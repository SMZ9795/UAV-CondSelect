"""Extract paired, frozen-model pooled activations. This entry performs inference only."""
import argparse,csv,json,math,os,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['rt','yolo'],required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--pairs',type=Path,default=Path(__file__).resolve().parents[1]/'data/response_pairs.csv');p.add_argument('--boxes',type=Path,default=Path(__file__).resolve().parents[1]/'data/measurement_boxes.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--yolo-weights',type=Path);a=p.parse_args()
 import numpy as np,torch
 import torch.nn.functional as F
 from PIL import Image
 root=Path(__file__).resolve().parents[1];rows=list(csv.DictReader(a.pairs.open()));boxes=json.loads(a.boxes.read_text());sources=sorted({r['source_id'] for r in rows});assert len(rows)==510 and len(sources)==102
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if a.model=='rt':
  sys.path.insert(0,str(root/'vendor/rtdetr'));from src.core import YAMLConfig
  model=YAMLConfig(str(a.config)).model;model.load_state_dict(ck['ema']['module'],strict=True);module=model.encoder;shape=(256,120,120)
 else:
  if not a.yolo_weights:p.error('--yolo-weights is required for model construction')
  os.environ.update(CALIBRATION_CONFIG=str(a.config.resolve()),UAV_YOLO_WEIGHTS=str(a.yolo_weights.resolve()))
  sys.path.insert(0,str(root/'src/yolo'));import common
  model,_=common.model(pretrained=False,device='cpu');model.load_state_dict(ck['ema'],strict=True)
  feature=json.loads(a.config.read_text())['feature'];module=model.get_submodule(feature['hook']);shape=tuple(feature['expected_shape'])
  assert feature['hook_output']=='output' and feature['region']=='global_background'
  assert feature['inference_dtype']=='float32' and feature['pool_dtype']=='float32'
  assert all(n*feature['stride']==d for n,d in zip(shape[-2:],feature['input_size']))
 grid=(960,960) if a.model=='rt' else tuple(feature['input_size'])
 threshold=0 if a.model=='rt' else feature['mask_threshold']
 kernel=9 if a.model=='rt' else feature['dilation_kernel']
 resize='nearest' if a.model=='rt' else feature['mask_resize']
 weight_resize='area' if a.model=='rt' else feature['weight_resize']
 epsilon=1e-12 if a.model=='rt' else feature['epsilon']
 assert kernel%2==1 and resize=='nearest' and weight_resize=='area'
 model=model.to(a.device).eval();captured=[]
 handle=module.register_forward_hook(lambda m,i,o:captured.append(o[0] if a.model=='rt' else o))
 def image_tensor(path):
  with Image.open(a.data_root/path) as im:v=np.asarray(im.convert('RGB').resize((grid[1],grid[0]),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255
  return torch.from_numpy(v.copy())[None].to(a.device)
 T=['rain','snow','fog','dust','low_light'];pooled=[];object_pooled=[]
 with torch.inference_mode():
  for sid in sources:
   rr={r['condition']:r for r in rows if r['source_id']==sid};r=rr['rain'];b=boxes[sid];W,H=b['size']
   with Image.open(a.data_root/r['protect_mask_path']) as im:prot=torch.from_numpy((np.asarray(im.convert('L'))>threshold).astype(np.float32))[None,None]
   aligned=F.interpolate(prot,size=grid,mode=resize);dilated=F.max_pool2d(aligned,kernel,1,kernel//2);support=1-dilated
   if a.model=='rt':
    x1,y1,x2,y2=b['bbox'];x1*=960/W;x2*=960/W;y1*=960/H;y2*=960/H;cx=(x1+x2)/2;cy=(y1+y2)/2;side=max(4*max(x2-x1,y2-y1),64)
    window=torch.zeros_like(support);window[:,:,math.floor(max(0,cy-side/2)):math.ceil(min(960,cy+side/2)),math.floor(max(0,cx-side/2)):math.ceil(min(960,cx+side/2))]=1;support*=window
   weight=F.interpolate(support,size=shape[-2:],mode=weight_resize)[0,0].to(a.device);assert weight.sum()>1e-8
   with Image.open(a.data_root/r['core_mask_path']) as im:core=torch.from_numpy((np.asarray(im.convert('L'))>0).astype(np.float32))[None,None]
   ow=F.interpolate(core,size=shape[-2:],mode='area')[0,0].to(a.device);pp=[];op=[]
   for path in [r['clean_path']]+[rr[t]['weather_path'] for t in T]:
    captured.clear();model(image_tensor(path));assert len(captured)==1;feat=captured[0][0];assert tuple(feat.shape)==shape
    pp.append(((feat*weight).sum((1,2))/(weight.sum()+epsilon)).cpu().numpy());op.append(((feat*ow).sum((1,2))/(ow.sum()+epsilon)).cpu().numpy())
   pooled.append(pp);object_pooled.append(op)
 handle.remove();a.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output,source_id=np.array(sources),condition=np.array(['clean']+T),pooled=np.array(pooled),object_pooled=np.array(object_pooled))
 print(a.output)
if __name__=='__main__':main()
