"""Install the archived response masks into a chosen data-root layout."""
import argparse,csv,hashlib,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--pairs',type=Path,default=Path(__file__).resolve().parents[1]/'data/response_pairs.csv');a=p.parse_args();seen=set()
 for r in csv.DictReader(a.pairs.open()):
  for key in ['core_mask_path','protect_mask_path']:
   dest=a.data_root/r[key]
   if dest in seen:continue
   seen.add(dest);src=a.assets/'masks'/key/(r['source_id']+'.png');expected=hashlib.sha256(src.read_bytes()).hexdigest()
   assert hashlib.sha256(src.read_bytes()).hexdigest()==expected
   if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==expected,'Existing mask differs: '+str(dest)
   else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 print('Verified or installed',len(seen),'masks.')
if __name__=='__main__':main()
