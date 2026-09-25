from common import *
from faster_coco_eval import COCO,COCOeval_faster
import torchvision  # native validation uses compiled torchvision NMS when imported
from ultralytics.utils.nms import non_max_suppression
import contextlib,io,gzip
DOM=['clean']+WEATHERS

def restore(d,im):
 d=d.detach().float().cpu().clone();assert torch.isfinite(d).all() and len(d)<=300
 d[:,[0,2]]*=im['width']/C['input_size'];d[:,[1,3]]*=im['height']/C['input_size']
 d[:,[0,2]]=d[:,[0,2]].clamp(0,im['width']);d[:,[1,3]]=d[:,[1,3]].clamp(0,im['height'])
 assert (d[:,5]==0).all()
 d[:,2:4]-=d[:,:2]
 return [{'image_id':im['id'],'category_id':1,'bbox':r[:4],'score':r[4]} for r in d.tolist() if min(r[2:4])>0]

def nms(pred):
 # One image per invocation; generous fixed time allowance avoids partial-batch truncation.
 return non_max_suppression(pred,conf_thres=0.,iou_thres=.7,max_det=300,nc=1,max_nms=30000,max_time_img=60.)

def coco_metrics(data,predictions):
 with contextlib.redirect_stdout(io.StringIO()):
  gt=COCO();gt.dataset=copy.deepcopy(data);gt.createIndex()
  if predictions:dt=gt.loadRes(copy.deepcopy(predictions))
  else:dt=COCO();dt.dataset={'images':data['images'],'categories':data['categories'],'annotations':[]};dt.createIndex()
  ev=COCOeval_faster(gt,dt,'bbox');ev.params.maxDets=[1,10,100];ev.params.imgIds=sorted(gt.imgs);ev.evaluate();ev.accumulate();ev.summarize()
 vals={k:(None if v<0 else float(v)*100) for k,v in zip(('AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl'),ev.stats)}
 assert all(vals[k] is not None and math.isfinite(vals[k]) for k in ['AP','AP50','AP75']);return vals

def state_sha(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

@torch.no_grad()
def evaluate(m,out,updates,smoke=False):
 before=rng();m.eval();data=read(C['validation_manifest']);images=sorted(data['images'],key=lambda x:x['id'])
 if smoke:images=[next(i for i in images if i['task']==d) for d in DOM]
 predictions=[];coverage=[];started=time.time();identity=state_sha(m)
 for im in images:
  output=m(tensor(im['file_name']).unsqueeze(0).cuda());detections=nms(output)[0];rows=restore(detections,im);predictions+=rows;coverage.append({'id':im['id'],'detections':len(rows)})
 assert len(coverage)==(6 if smoke else 220) and len({r['id'] for r in coverage})==len(images)
 metrics={}
 for domain in DOM:
  ims=[i for i in images if i['task']==domain];ids={i['id'] for i in ims};subset={**data,'images':ims,'annotations':[a for a in data['annotations'] if a['image_id'] in ids]}
  metrics[domain]=coco_metrics(subset,[p for p in predictions if p['image_id'] in ids])
 out=Path(out);out.mkdir(parents=True,exist_ok=True);p=out/f'predictions_{updates:05}.json.gz';tmp=p.with_suffix('.tmp')
 with gzip.open(tmp,'wt',compresslevel=6) as f:json.dump(predictions,f,separators=(',',':'),allow_nan=False)
 tmp.replace(p)
 rec={'updates':updates,'primary':sum(v['AP'] for v in metrics.values())/6,'domains':metrics,'coverage':coverage,'ema_sha256':identity,'manifest_sha256':sha(C['validation_manifest']),'predictions_sha256':sha(p),'seconds':time.time()-started,'smoke':smoke}
 js(out/f'metrics_{updates:05}.json',rec);restore_rng(before);return rec
