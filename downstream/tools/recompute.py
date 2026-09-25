"""Recompute response matrices, selection, AP aggregates and transfer statistics on CPU."""
from pathlib import Path
import argparse,csv,json,itertools,collections
import numpy as np
from scipy.stats import pearsonr,spearmanr
T=['rain','snow','fog','dust','low_light']
def normname(s):return s.replace('dim_low_light','low_light')
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def objective(S,A,lam):
 F=float(S[:,A].max(axis=1).sum());R=float(sum(S[a,b] for a,b in itertools.combinations(A,2)))
 return {'F':F,'R':R,'J':F-lam*R}
def select(S,lam):
 A=list(range(5));steps=[];deleted=[]
 while len(A)>1:
  before=objective(S,A,lam);candidates=[]
  for a in A:
   after=objective(S,[i for i in A if i!=a],lam)
   candidates.append({'condition':T[a],'index':a,**after,'gain':after['J']-before['J']})
  best=max(candidates,key=lambda x:x['gain'])
  steps.append({'retained':[T[a] for a in A],**before,'candidates':candidates,'deleted':best['condition'] if best['gain']>0 else None})
  if best['gain']<=0:break
  A.remove(best['index']);deleted.append(best['condition'])
 return {'lambda':lam,'steps':steps,'deleted':deleted,'retained':[T[a] for a in A],'final':objective(S,A,lam),'stop':'singleton' if len(A)==1 else 'no strictly positive deletion gain'}
def responses(assets,out):
 matrices={};checks={}
 z=np.load(assets/'responses/rt.npz');assert [normname(x) for x in (z['weather'] if 'weather' in z else z['condition'][1:])]==T and len(set(z['source_id']))==102
 for region in ['context','object']:
  key=region+'_L_high'
  if 'h_weather_'+key in z:delta=z['h_weather_'+key]-z['h_clean_'+key][:,None]
  else:
   pooled=z['pooled' if region=='context' else 'object_pooled'];delta=pooled[:,1:]-pooled[:,0:1]
  u=np.asarray([[v/(float(np.linalg.norm(v))+1e-12) for v in source] for source in delta],dtype=np.float32)
  # Preserve the recorded FP32 arithmetic and mean reduction.
  similarities=np.stack([np.stack([np.sum(u[:,a]*u[:,b],axis=1) for b in range(5)],axis=1) for a in range(5)],axis=1)
  S=np.asarray([[float(similarities[:,a,b].mean()) for b in range(5)] for a in range(5)]);checks['rt_'+region+'_cache_max_error']=float(np.max(np.abs(similarities-z['per_source_similarity_'+key]))) if 'per_source_similarity_'+key in z else None
  measured=S.astype(float);selection=measured.copy();np.fill_diagonal(selection,1)
  matrices['rt_'+region]=selection
  save(out/f'rt_{region}_measured.json',{'order':T,'source_id':z['source_id'].tolist(),'matrix':measured.tolist(),'precision':'FP32 pooled delta, normalization, per-source dot and mean'})
  save(out/f'rt_{region}_selection.json',{'order':T,'matrix':selection.tolist()})
 z=np.load(assets/'responses/yolo.npz');assert [normname(x) for x in z['condition']]==['clean']+T
 delta=z['pooled'][:,1:]-z['pooled'][:,0:1];n=np.linalg.norm(delta.astype(np.float64),axis=-1)
 u=delta.astype(np.float64)/(n[...,None]+1e-12)
 sim=np.empty((102,5,5))
 for a,b in itertools.product(range(5),repeat=2):sim[:,a,b]=(u[:,a]*u[:,b]).sum(-1)
 S=np.array([[sim[:,a,b].mean() for b in range(5)] for a in range(5)]);checks['yolo_cache_max_error']=float(np.max(np.abs(sim-z['per_source_cosine']))) if 'per_source_cosine' in z else None
 save(out/'yolo_context_measured.json',{'order':T,'source_id':z['source_id'].tolist(),'matrix':S.tolist(),'precision':'FP32 delta; FP64 stabilized normalization, dot and mean'})
 np.fill_diagonal(S,1);matrices['yolo_context']=S
 save(out/'yolo_context_selection.json',{'order':T,'matrix':S.tolist()})
 for name,lambdas in [('rt_context',[0,.3,.6,.8,2,4,6]),('yolo_context',[0,.1,.3,1])]:
  S=matrices[name];assert np.allclose(S,S.T,atol=1e-12,rtol=0) and np.array_equal(S.diagonal(),np.ones(5))
  save(out/(name+'_selection_traces.json'),[select(S,x) for x in lambdas])
 save(out/'response_checks.json',checks);return matrices

def aggregates(inputs,out):
 runs=read(inputs/'runs.json');groups=collections.defaultdict(list);flat=[]
 for r in runs:
  components=r['components_AP'];assert len(components)==(6 if r['protocol']=='internal' else 2)
  ap=float(np.mean(list(components.values())));assert abs(ap-r['aggregate_AP'])<1e-10
  subset=''.join('RSFDL'[i] for i,t in enumerate(T) if t in [normname(x) for x in r['conditions']]) or 'Clean-only'
  groups[r['model'],r['protocol'],subset].append((r['seed'],ap));flat.append({'model':r['model'],'protocol':r['protocol'],'subset':subset,'seed':r['seed'],'AP':ap})
 summary=[]
 for (model,protocol,subset),values in groups.items():
  assert sorted(s for s,a in values)==[0,1,2];a=[a for s,a in values];summary.append({'model':model,'protocol':protocol,'subset':subset,'mean_AP':float(np.mean(a)),'sample_SD':float(np.std(a,ddof=1))})
 order=['Clean-only']+[''.join(c) for k in range(1,6) for c in itertools.combinations('RSFDL',k)]
 key=lambda r:(r['model'],r['protocol'],order.index(r['subset']))
 flat.sort(key=lambda r:(*key(r),r['seed']));summary.sort(key=key)
 table(out/'AP_per_seed.csv',flat);table(out/'AP_summary.csv',summary)
 for model,protocol in sorted({(r['model'],r['protocol']) for r in summary}):
  group=[r for r in summary if (r['model'],r['protocol'])==(model,protocol)];assert len(group)==32
  rows=[];lines=['| Subset | Seed | AP | Sample SD |','|---|---|---:|---:|']
  for r in group:
   for q in [q for q in flat if key(q)==key(r)]:
    rows.append({'subset':r['subset'],'seed':q['seed'],'AP':q['AP'],'sample_SD':''})
    lines.append(f"| {r['subset']} | {q['seed']} | {q['AP']:.2f} | |")
   rows.append({'subset':r['subset'],'seed':'mean','AP':r['mean_AP'],'sample_SD':r['sample_SD']})
   lines.append(f"| {r['subset']} | mean | {r['mean_AP']:.2f} | {r['sample_SD']:.2f} |")
  stem=('rt' if model=='RT-DETRv2-S' else 'yolo')+'_'+protocol
  table(out/(stem+'.csv'),rows);save(out/(stem+'.json'),rows)
  (out/(stem+'.md')).write_text('\n'.join(lines)+'\n')
 ratio=[]
 for r in read(inputs/'ratio_runs.json'):
  c=r['components_AP'];a=(c['weather130']+c['Swarm117'])/2;assert abs(a-r['aggregate_AP'])<1e-10;ratio.append({'configuration':r['configuration'],'seed':r['seed'],'real_AP':c['weather130'],'synthetic_AP':c['Swarm117'],'AP':a})
 table(out/'ratio_per_seed.csv',ratio)
 table(out/'ratio_summary.csv',[{'configuration':tag,'mean_AP':float(np.mean([r['AP'] for r in ratio if r['configuration']==tag])),'sample_SD':float(np.std([r['AP'] for r in ratio if r['configuration']==tag],ddof=1))} for tag in sorted({r['configuration'] for r in ratio})])

def transfer(inputs,out,matrices):
 rr=read(inputs/'transfer_AP.json');assert len(rr)==90
 ap={(r['seed'],normname(r['arm']),normname(r['target'])):100*r['AP'] for r in rr};seeds=[1,2,3];directed=[];symmetric=[];cor=[]
 for seed in seeds:
  for a,b in itertools.permutations(T,2):directed.append({'seed':seed,'a':a,'b':b,'AP':ap[seed,a,b],'control_AP':ap[seed,'cleanx2',b],'gain':ap[seed,a,b]-ap[seed,'cleanx2',b]})
  for a,b in itertools.combinations(T,2):symmetric.append({'seed':seed,'a':a,'b':b,'gain':((ap[seed,a,b]-ap[seed,'cleanx2',b])+(ap[seed,b,a]-ap[seed,'cleanx2',a]))/2})
 for label,rows in [('directed',directed),('symmetric',symmetric)]:
  pairs=list(itertools.permutations(T,2) if label=='directed' else itertools.combinations(T,2));means=[]
  for a,b in pairs:
   values=[r['gain'] for r in rows if (r['a'],r['b'])==(a,b)];means.append({'a':a,'b':b,'mean_gain':float(np.mean(values)),'sample_SD':float(np.std(values,ddof=1))})
  table(out/f'transfer_{label}_per_seed.csv',rows);table(out/f'transfer_{label}_mean.csv',means)
  for region in ['context','object']:
   x=[matrices['rt_'+region][T.index(a),T.index(b)] for a,b in pairs]
   for seed in ['mean']+seeds:
    y=[r['mean_gain'] for r in means] if seed=='mean' else [next(r['gain'] for r in rows if (r['seed'],r['a'],r['b'])==(seed,a,b)) for a,b in pairs]
    cor.append({'region':region,'transfer':label,'seed':seed,'n':len(x),'Spearman':float(spearmanr(x,y).statistic),'Pearson':float(pearsonr(x,y).statistic)})
 table(out/'transfer_correlations.csv',cor)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path);p.add_argument('--results-only',action='store_true');p.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parents[1]/'results');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if a.results_only:
  aggregates(a.inputs,a.output);print('Exported AP results in fixed configuration order.');return
 if a.assets is None:p.error('--assets is required unless --results-only is set')
 matrices=responses(a.assets,a.output);aggregates(a.inputs,a.output);transfer(a.inputs,a.output,matrices)
 print('Recomputed matrices, traces, 288 main runs, 12 ratio runs and 90 transfer AP records.')
if __name__=='__main__':main()
