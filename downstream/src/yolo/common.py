from pathlib import Path
import os
R=Path(os.environ.get("DOWNSTREAM_ROOT",Path(__file__).resolve().parents[1]))
REFERENCE=Path(os.environ['UAV_YOLO_WEIGHTS'])
os.environ.setdefault('YOLO_CONFIG_DIR',str(R/'configs/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(R/'tmp/matplotlib'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
import json, hashlib, random, math, shutil, subprocess, time, copy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader,Sampler
from types import SimpleNamespace
CONFIG=Path(os.environ['CALIBRATION_CONFIG']);C=json.loads(CONFIG.read_text())
WEATHERS=['rain','snow','fog','dust','dim_low_light']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def space(extra=0):
 v=os.statvfs(R)
 if v.f_bavail*v.f_frsize<C['storage']['minimum_free_bytes']+extra or v.f_favail<C['storage']['minimum_free_inodes']:raise RuntimeError('Insufficient checkpoint storage reserve')
 return {'free_bytes':v.f_bavail*v.f_frsize,'free_inodes':v.f_favail}

def atomic_save(p,d):
 space(800*1024**2);p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('wb') as f:torch.save(d,f);f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def seed():
 torch.set_num_threads(4);random.seed(C['seed']);np.random.seed(C['seed']);torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed'])
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(x):
 random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda'])
def tensor(path):
 with Image.open(path) as im:a=np.asarray(im.convert('RGB').resize((C['input_size'],C['input_size']),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255.
 return torch.from_numpy(a.copy())
def normalized_boxes(im,annotations):
 """COCO xywh -> clamped, sanitized YOLO normalized cxcywh; empty is (0,4)."""
 W,H=im['width'],im['height'];assert W>0 and H>0;b=[]
 for a in annotations:
  assert a['category_id']==1
  if a.get('iscrowd',0):continue
  x,y,w,h=map(float,a['bbox'])
  if not all(math.isfinite(v) for v in (x,y,w,h)):raise ValueError('Nonfinite bbox')
  x1,y1=max(0.,min(W,x)),max(0.,min(H,y))
  x2,y2=max(0.,min(W,x+w)),max(0.,min(H,y+h))
  w,h=x2-x1,y2-y1
  if min(w*C['input_size']/W,h*C['input_size']/H)<1:continue
  b.append([(x1+w/2)/W,(y1+h/2)/H,w/W,h/H])
 return torch.tensor(b,dtype=torch.float32).reshape(-1,4)
class CleanDataset(Dataset):
 def __init__(self):
  a=read(R/C['training_manifest']);self.images=sorted(a['images'],key=lambda x:x['id']);self.anns={i['id']:[] for i in self.images}
  for x in a['annotations']:self.anns[x['image_id']].append(x)
 def __len__(self):return len(self.images)
 def __getitem__(self,k):
  im=self.images[k]
  return tensor(im['file_name']),normalized_boxes(im,self.anns[im['id']]),im['base_source_identity']
def collate(items):
 return {'img':torch.stack([i[0] for i in items]),'bboxes':torch.cat([i[1] for i in items]),'cls':torch.zeros(sum(len(i[1]) for i in items),1),'batch_idx':torch.cat([torch.full((len(i[1]),),k) for k,i in enumerate(items)]),'sources':[i[2] for i in items]}
class RepeatSampler(Sampler):
 def __init__(self,start=0,dataset_size=None):
  self.start=start;self.perms={};self.dataset_size=len(CleanDataset()) if dataset_size is None else dataset_size
  assert self.dataset_size>0 and start>=0 and C['physical_batch']>0
 def batch(self,k):
  out=[];pos=k*C['physical_batch']
  while len(out)<C['physical_batch']:
   epoch,offset=divmod(pos,self.dataset_size)
   if epoch not in self.perms:
    a=list(range(self.dataset_size));h=hashlib.sha256(('calibration-uniform-order\0'+str(C['seed'])+'\0'+str(epoch)).encode()).digest();random.Random(int.from_bytes(h[:8],'big')).shuffle(a);self.perms={epoch:a}
   take=min(C['physical_batch']-len(out),self.dataset_size-offset);out.extend(self.perms[epoch][offset:offset+take]);pos+=take
  return out
 def __iter__(self):
  k=self.start
  while True:yield self.batch(k);k+=1
 def __len__(self):return 2**31-1
def loader(start=0):return DataLoader(CleanDataset(),batch_sampler=RepeatSampler(start),collate_fn=collate,num_workers=C['workers'],prefetch_factor=C['prefetch_factor'],pin_memory=True,generator=torch.Generator().manual_seed(C['seed']))
def model(pretrained=True,device='cuda'):
 from ultralytics.nn.tasks import DetectionModel
 from ultralytics.utils import DEFAULT_CFG_DICT
 p=REFERENCE;assert sha(p)=='d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'
 w=torch.load(p,map_location='cpu',weights_only=False)['model'].float();cfg=copy.deepcopy(w.yaml);cfg['scale']='m'
 m=DetectionModel(cfg,ch=3,nc=1,verbose=False)
 source=w.state_dict();dest=m.state_dict();matched={k:v for k,v in source.items() if k in dest and v.shape==dest[k].shape}
 report={'matched_tensors':len(matched),'mismatches':[k for k in source if k in dest and source[k].shape!=dest[k].shape],'nc':m.model[-1].nc,'strides':m.stride.tolist(),'parameters':sum(p.numel() for p in m.parameters())}
 if pretrained:m.load_state_dict(matched,strict=False)
 args=dict(DEFAULT_CFG_DICT);args.update({k:v for k,v in C['loss'].items() if k!='reduction'});args.update(C['augmentations']);args.update({'imgsz':960,'cache':False,'single_cls':True});m.args=SimpleNamespace(**args);m.names={0:'UAV'}
 for n,p in m.named_parameters():p.requires_grad_('.dfl.' not in n)
 assert m.model[-1].nc==1 and m.model[-1].f==[16,19,22] and m.stride.tolist()==[8,16,32]
 m=m.to(device)
 from stable_ciou import install
 install(m)
 return m,report
def optimizer(m):
 groups={'decay':[],'norm':[],'bias':[]};names={k:[] for k in groups}
 for mn,mod in m.named_modules():
  for pn,p in mod.named_parameters(recurse=False):
   if not p.requires_grad:continue
   g='bias' if pn=='bias' else 'norm' if isinstance(mod,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.LayerNorm,torch.nn.GroupNorm)) else 'decay';groups[g].append(p);names[g].append(mn+'.'+pn)
 assert sum(len(x) for x in groups.values())==sum(p.requires_grad for p in m.parameters())
 o=torch.optim.AdamW([{'params':v,'weight_decay':C['weight_decay'] if k=='decay' else 0.,'group_name':k} for k,v in groups.items()],lr=C['lr'],betas=tuple(C['betas']),eps=C['eps'],foreach=False,fused=False)
 return o,names
class Schedule:
 def __init__(self,o):self.o=o;self.completed=0
 def prepare(self):
  u=self.completed+1
  factor=u/C['warmup_updates'] if u<=C['warmup_updates'] else C['final_lr_factor']+(1-C['final_lr_factor'])*(1+math.cos(math.pi*(u-C['warmup_updates'])/(C['schedule_horizon']-C['warmup_updates'])))/2
  lr=C['lr']*factor
  for g in self.o.param_groups:g['lr']=lr
  return lr
 def state_dict(self):return {'completed':self.completed,'base_lr':C['lr'],'warmup':C['warmup_updates'],'horizon':C['schedule_horizon'],'final_factor':C['final_lr_factor']}
 def load_state_dict(self,d):
  expect=self.state_dict();expect['completed']=d['completed'];assert d==expect;self.completed=d['completed']
def identities():return {'config':sha(CONFIG),'train':sha(C['training_manifest']),'validation':sha(C['validation_manifest']) if C.get('validation_manifest') else None,'pretrained':sha(REFERENCE),'code':{p.name:sha(p) for p in sorted((Path(__file__).resolve().parent).glob('*.py'))}}
class Engine:
 def __init__(self,checkpoint=None):
  from ultralytics.utils.torch_utils import ModelEMA
  seed();self.m,self.preload=model();self.o,self.groups=optimizer(self.m);self.scheduler=Schedule(self.o);self.ema=ModelEMA(self.m,decay=C['ema_decay'],tau=C['ema_tau']);self.scaler=torch.amp.GradScaler('cuda',init_scale=C['initial_scale']);self.step_calls=0
  self.o.register_step_post_hook(lambda *a:setattr(self,'step_calls',self.step_calls+1))
  self.counters={'updates':0,'microbatches':0,'attempts':0,'skips':0,'exposures':0};self.identity=identities()
  if checkpoint:
   d=torch.load(checkpoint,map_location='cpu',weights_only=False)
   from recovery_guard import require_complete_boundary
   require_complete_boundary(d)
   assert d['identities']==self.identity
   assert d['config']==C,'Runtime settings changed on resume'
   assert d['sampler']=={'next_microbatch':d['counters']['microbatches'],'physical_batch':C['physical_batch'],'accumulation':C['accumulation']}
   self.m.load_state_dict(d['model']);self.o.load_state_dict(d['optimizer']);self.ema.ema.load_state_dict(d['ema']);self.ema.updates=d['ema_updates'];self.scaler.load_state_dict(d['scaler']);self.scheduler.load_state_dict(d['scheduler']);self.counters=d['counters'];restore_rng(d['rng'])
  assert self.scheduler.completed==self.counters['updates']==self.ema.updates
  self.m.train();self.it=iter(loader(self.counters['microbatches']))
 def payload(self):return {'model':self.m.state_dict(),'optimizer':self.o.state_dict(),'ema':self.ema.ema.state_dict(),'ema_updates':self.ema.updates,'scaler':self.scaler.state_dict(),'scheduler':self.scheduler.state_dict(),'counters':self.counters.copy(),'rng':rng(),'identities':self.identity,'sampler':{'next_microbatch':self.counters['microbatches'],'physical_batch':C['physical_batch'],'accumulation':C['accumulation']},'config':C}
 def step(self):
  t=time.perf_counter();lr=self.scheduler.prepare();self.o.zero_grad(set_to_none=True);sources=[];losses=[];components=[]
  for _ in range(C['accumulation']):
   b=next(self.it);sources+=b.pop('sources');b={k:v.cuda(non_blocking=True) for k,v in b.items()};self.counters['microbatches']+=1;self.counters['exposures']+=len(b['img'])
   # Native v8DetectionLoss already multiplies by the physical batch size.
   # Native trainer sums microbatch gradients; an additional /accumulation changes that scale.
   with torch.autocast('cuda',dtype=torch.float16):lv,comp=self.m(b);loss=lv.sum()
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   self.scaler.scale(loss).backward();losses.append(float(loss.detach()));components.append(comp.detach().float().cpu().tolist())
  self.scaler.unscale_(self.o);gn=float(torch.nn.utils.clip_grad_norm_(self.m.parameters(),C['clip_grad_norm']));old=self.step_calls;scale=self.scaler.get_scale();self.scaler.step(self.o);self.scaler.update();ok=self.step_calls>old;self.o.zero_grad(set_to_none=True);self.counters['attempts']+=1
  if ok:self.counters['updates']+=1;self.scheduler.completed+=1;self.ema.update(self.m)
  else:self.counters['skips']+=1
  torch.cuda.synchronize();return {**self.counters,'successful':ok,'loss':sum(losses),'components':components,'lr':lr,'grad_norm':gn if math.isfinite(gn) else None,'scale_before':scale,'scale_after':self.scaler.get_scale(),'sources':sources,'seconds':time.perf_counter()-t,'gpu_peak_allocated':torch.cuda.max_memory_allocated(),'gpu_peak_reserved':torch.cuda.max_memory_reserved()}
