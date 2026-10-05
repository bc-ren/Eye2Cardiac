"""Fresh student-dependent features at the inherited disease CFP visits."""
import argparse,time,multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd,torch
from runlib import *
sys.path.insert(0,str(STUDENT))
import shared as ss
from student import make_models,args_of
from train_adapter import cfg
from data import device_batch,worker_init
from objectives import describe
from evaluate_reconstruction import init_geometry,geometry_job
from torch.utils.data import DataLoader,Subset

def init_geometry_limited():
 global THREAD_LIMIT
 from threadpoolctl import threadpool_limits
 THREAD_LIMIT=threadpool_limits(C['geometry_blas_threads'])
 torch.set_num_threads(C['geometry_blas_threads'])
 init_geometry()

def main(seed,smoke):
 torch.set_num_threads(2);run=run_name(seed);folder=STUDENT/'runs'/run
 a=torch.load(folder/'sync_only_best.pt',map_location='cpu',weights_only=False)
 c=ss.relocated_config(a['config'],cfg());m,dec,_=make_models(c);m.restore(a['model']);m.eval();m.requires_grad_(False);dec.requires_grad_(False)
 ref=module('inference_reference',HERE/'clinical_dataset.py');ref.PREPARED=ss.PREP
 d=pd.read_parquet(manifest('Diagnosis'))
 if smoke:d=d[d.original_split.eq('train')&d.pre_eligible].iloc[:8].reset_index(drop=True)
 with stage(ROOT/('feature_smoke' if smoke else 'features')/str(seed),seed=seed) as out,ProcessPoolExecutor(max_workers=C['geometry_workers'],mp_context=mp.get_context('spawn'),initializer=init_geometry_limited) as pool,torch.inference_mode():
  ds=ref.Clinical(d)
  arrays={k:np.lib.format.open_memmap(out/(k+'_SERVER_ONLY.npy'),mode='w+',dtype='float32',shape=(len(d),n)) for k,n in [('z0',96),('retina',1024),('mesh11',11)]}
  d[['eid','retina_date','age_cfp','sex_cfp','bsa_cfp']].to_parquet(out/'manifest_SERVER_ONLY.parquet',index=False)
  start=time.time();done=0;invalid=0;resume=None
  initial_done=done;last_saved=done;pending=deque();gpu_work_seconds=0.;geometry_wait_seconds=0.
  dl=DataLoader(Subset(ds,range(done,len(d))),batch_size=C['feature_batch'],num_workers=4,pin_memory=True,persistent_workers=True,worker_init_fn=worker_init)
  def consume():
   nonlocal done,invalid,last_saved,geometry_wait_seconds
   ix,basics,futures,bad=pending.popleft();t=time.perf_counter()
   result=[f.result() for f in futures];geometry_wait_seconds+=time.perf_counter()-t
   ph=np.array([[r[f] for f in FIELDS11] for r in result],np.float32)
   assert np.isfinite(ph).all();np.testing.assert_allclose(ph[:,:5],basics,rtol=3e-5,atol=.01)
   np.testing.assert_array_equal(ix,np.arange(done,done+len(ix)))
   arrays['mesh11'][ix]=ph;done+=len(ix);invalid+=bad
   if done-last_saved>=512 or done==len(d):
    for x in arrays.values():x.flush()
    last_saved=done
    save(out/'PROGRESS.json',dict(done=done,total=len(d),resumed_n=initial_done,seconds=time.time()-start,patients_per_second=(done-initial_done)/max(time.time()-start,1e-9),invalid_mesh_n=invalid,gpu_work_wall_seconds=gpu_work_seconds,geometry_wait_seconds=geometry_wait_seconds));print(json.dumps(dict(done=done,total=len(d))),flush=True)
  for batch in dl:
   t=time.perf_counter()
   ix=batch['row_index'].numpy();batch=device_batch(batch,'cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):o=m(**args_of(batch))
   for k in ['z0','retina']:arrays[k][ix]=o[k].float().cpu().numpy()
   jobs=[];basics=[];bad=0
   for start_ in range(0,len(ix),C['decode_batch']):
    mesh=dec(o['z0'][start_:start_+C['decode_batch']]);v=describe(mesh,dec);rr=torch.arange(len(mesh),device='cuda');bad+=int(v['invalid'].sum())
    pair=torch.stack([mesh[rr,v['ed']],mesh[rr,v['es']]],1).cpu().numpy();basics.append(v['ph'].cpu().numpy())
    for j,idx in enumerate(ix[start_:start_+C['decode_batch']]):jobs.append((int(idx),str(d.iloc[idx].eid),pair[j],int(v['ed'][j]),int(v['es'][j])))
   gpu_work_seconds+=time.perf_counter()-t
   pending.append((ix,np.concatenate(basics),[pool.submit(geometry_job,j) for j in jobs],bad))
   if len(pending)>=C['pending_feature_batches']:consume()
  while pending:consume()
  assert done==len(d)
  for x in arrays.values():x.flush();assert np.isfinite(x).all()
  save(out/'PROVENANCE.json',dict(n=len(d),student_run=run,checkpoint_sha256=sha(folder/'sync_only_best.pt'),designated_phase='sync_only',retizero_last4_finetuned=True,learned_features_reused=False,source_manifest_sha256=sha(manifest('Diagnosis')),age_anchor='exact CFP visit',future_CMR_or_diagnosis_inputs=False,time_transport_used=False,fields11=FIELDS11,excluded_before_head_preprocessing=C['exclude'],invalid_mesh_n=invalid,resumed_prefix=resume,execution_changes_only=True,outputs={k:sha(out/(k+'_SERVER_ONLY.npy')) for k in arrays}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.seed,a.smoke)
