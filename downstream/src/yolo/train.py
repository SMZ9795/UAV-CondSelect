"""Standalone wrapper around the recorded YOLO Engine."""
from common import *

def main():
 out=R/'run';out.mkdir(parents=True,exist_ok=False);e=Engine()
 js(out/'optimizer_groups.json',e.groups);js(out/'pretrained_load.json',e.preload)
 with (out/'metrics.jsonl').open('w') as f:
  while e.counters['updates']<C['successful_updates']:
   rec=e.step();f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush()
   if rec['successful'] and rec['updates']%C['checkpoint_interval']==0:atomic_save(out/'last.pt',e.payload())
 atomic_save(out/'last.pt',e.payload());atomic_save(out/'final_ema.pt',{'ema':e.ema.ema.state_dict(),'updates':e.counters['updates'],'identities':e.identity,'config':C})
 js(out/'COMPLETE.json',{'counters':e.counters,'weights':'final EMA','scheduler':e.scheduler.state_dict()})
if __name__=='__main__':main()
