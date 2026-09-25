"""Plot measured matrices and transfer means with sample-SD error bars."""
import argparse,json,csv
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 a.output.mkdir(parents=True,exist_ok=True)
 for name in ['rt_context','yolo_context']:
  d=json.loads((a.results/(name+'_measured.json')).read_text());S=np.array(d['matrix']);fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(S,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(5),d['order'],rotation=35,ha='right');ax.set_yticks(range(5),d['order'])
  for i in range(5):
   for j in range(5):ax.text(j,i,f'{S[i,j]:.3f}',ha='center',va='center')
  fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(a.output/(name+'.pdf'));plt.close(fig)
 d=json.loads((a.results/'rt_context_measured.json').read_text());S=np.array(d['matrix']);T=d['order'];rows=list(csv.DictReader((a.results/'transfer_symmetric_mean.csv').open()));x=[S[T.index(r['a']),T.index(r['b'])] for r in rows];y=[float(r['mean_gain']) for r in rows];sd=[float(r['sample_SD']) for r in rows]
 fig,ax=plt.subplots();ax.errorbar(x,y,yerr=sd,fmt='o');ax.set(xlabel='Context response similarity',ylabel='Symmetric transfer gain (AP points)');fig.tight_layout();fig.savefig(a.output/'transfer.pdf');plt.close(fig)
if __name__=='__main__':main()
