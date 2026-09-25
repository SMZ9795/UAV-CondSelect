"""Evaluate one saved EMA checkpoint and one COCO dataset, retaining predictions."""
import argparse,json,sys,os,hashlib,copy
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['rt','yolo'],required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--annotation',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--yolo-weights',type=Path);p.add_argument('--device',default='cuda');a=p.parse_args()
 import numpy as np,torch
 from PIL import Image
 from faster_coco_eval import COCO,COCOeval_faster
 root=Path(__file__).resolve().parents[1];data=json.loads(a.annotation.read_text());ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if a.model=='rt':
  sys.path.insert(0,str(root/'vendor/rtdetr'));from src.core import YAMLConfig
  cfg=YAMLConfig(str(a.config));model=cfg.model;model.load_state_dict(ck['ema']['module'],strict=True);post=cfg.postprocessor.to(a.device).eval()
 else:
  if not a.yolo_weights:p.error('--yolo-weights is required')
  os.environ.update(CALIBRATION_CONFIG=str(a.config.resolve()),UAV_YOLO_WEIGHTS=str(a.yolo_weights.resolve()));sys.path.insert(0,str(root/'src/yolo'));from common import model as build
  from validation import nms,restore
  model,_=build(pretrained=False,device='cpu');model.load_state_dict(ck['ema'],strict=True)
 model=model.to(a.device).eval();pred=[]
 with torch.inference_mode():
  for im in sorted(data['images'],key=lambda i:i['id']):
   with Image.open(a.data_root/im['file_name']) as img:x=np.asarray(img.convert('RGB').resize((960,960),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255
   output=model(torch.from_numpy(x.copy())[None].to(a.device))
   if a.model=='yolo':pred.extend(restore(nms(output)[0],im))
   else:
    result=post(output,torch.tensor([[im['width'],im['height']]],device=a.device))[0];boxes=result['boxes'].cpu();boxes[:,2:]-=boxes[:,:2]
    assert len(boxes)==300 and set(result['labels'].tolist())=={1}
    pred.extend({'image_id':im['id'],'category_id':int(label),'bbox':box,'score':score} for box,label,score in zip(boxes.tolist(),result['labels'].tolist(),result['scores'].tolist()))
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'predictions.json').write_text(json.dumps(pred)+'\n')
 domains=sorted({im.get('task','pooled') for im in data['images']});metrics={}
 for domain in domains:
  ids={im['id'] for im in data['images'] if im.get('task','pooled')==domain};gt=COCO();gt.dataset={**data,'images':[i for i in data['images'] if i['id'] in ids],'annotations':[v for v in data['annotations'] if v['image_id'] in ids]};gt.createIndex();pp=[v for v in pred if v['image_id'] in ids]
  if pp:dt=gt.loadRes(pp)
  else:dt=COCO();dt.dataset={**gt.dataset,'annotations':[]};dt.createIndex()
  ev=COCOeval_faster(gt,dt,'bbox');ev.params.imgIds=sorted(ids);ev.params.maxDets=[1,10,100];ev.evaluate();ev.accumulate();ev.summarize();metrics[domain]={k:float(v)*100 for k,v in zip(['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl'],ev.stats)}
 (a.output/'metrics.json').write_text(json.dumps({'components':metrics,'equal_domain_mean_AP':float(np.mean([v['AP'] for v in metrics.values()])),'weights':'EMA','maxDets':[1,10,100],'units':'0-100'},indent=2)+'\n')
if __name__=='__main__':main()
