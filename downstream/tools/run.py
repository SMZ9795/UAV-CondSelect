"""Portable training entry. Run with --prepare-only for a CPU-only configuration check."""
import argparse,os,subprocess,sys,json
from pathlib import Path
from materialize import render

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('family',choices=['rt','ratio','yolo','transfer']);p.add_argument('--config',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 root=Path(__file__).resolve().parents[1];out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True);config=render(a.config,a.data_root,out,a.weights)
 env=dict(os.environ,UAV_DATA_ROOT=str(a.data_root.resolve()),UAV_YOLO_WEIGHTS=str(a.weights.resolve()),DOWNSTREAM_ROOT=str(out),CALIBRATION_CONFIG=str(config),TMPDIR=str(out/'tmp'),PYTHONDONTWRITEBYTECODE='1');(out/'tmp').mkdir(exist_ok=True)
 if a.family=='yolo':cmd=[sys.executable,str(root/'src/yolo/train.py')]
 elif a.family=='transfer':cmd=[sys.executable,str(root/'src/transfer/train_steps.py'),'--config',str(config),'--dataset-root',str(a.data_root.resolve()/'rtdetrv2_pilot_dataset'),'--output-dir',str(out/'run')]
 else:cmd=[sys.executable,str(root/'src/rt'/('train_ratio.py' if a.family=='ratio' else 'train.py')),'--config',str(config),'--output-dir',str(out/'run')]
 print(json.dumps({'configuration':str(config),'command':cmd,'prepare_only':a.prepare_only},indent=2))
 if not a.prepare_only:subprocess.run(cmd,env=env,check=True)
if __name__=='__main__':main()
