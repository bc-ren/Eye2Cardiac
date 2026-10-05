import argparse,pickle,time as clock
import numpy as np,pandas as pd,torch
from scipy.special import expit
from runlib import *
from mlp import train,predict,threshold,ClinicalScore,fit_baseline,risks
M=module('metrics',HERE/'metrics.py')
def harrell(t,y,r):return M.cindex(M.cargs(t,np.asarray(y,bool),r),np.ones(len(y)))
def task_data(task,variant):
 d=pd.read_parquet(manifest(task));pre=task=='Diagnosis'
 y=d.pre_y.to_numpy(bool) if pre else d.event.to_numpy(bool)
 time=None if pre else d.time_years.to_numpy(float)
 eligible=d.pre_eligible.to_numpy(bool) if pre else d.incident_eligible.to_numpy(bool)
 dev=np.flatnonzero(d.original_split.isin(['train','validation']).to_numpy()&eligible)
 fid=d['pre_fold' if pre else 'surv_fold'].to_numpy(int)
 role=d.matched_augmentation_role
 aug=np.flatnonzero(role.eq('positive_50pct').to_numpy()) if variant=='positive_only' else np.flatnonzero(role.ne('none').to_numpy()) if variant=='matched_1to5' else np.array([],int)
 assert np.all(eligible[aug]);folds=[(np.r_[dev[fid[dev]!=j],aug],dev[fid[dev]==j]) for j in range(5)]
 return d,y,time,dev,aug,folds

def main(seed,task,variant):
 torch.set_num_threads(4);pre=task=='Diagnosis';assert state(ROOT/'features'/str(seed))=='COMPLETED';assert state(ROOT/'clinical_raw')=='COMPLETED'
 d,y,time,dev,aug,folds=task_data(task,variant);final=np.r_[dev,aug];allidx=np.arange(len(d))
 fp=ROOT/'features'/str(seed);fd=pd.read_parquet(fp/'manifest_SERVER_ONLY.parquet')
 assert fd.eid.astype(str).tolist()==d.eid.astype(str).tolist();assert pd.to_datetime(fd.retina_date).equals(pd.to_datetime(d.retina_date))
 np.testing.assert_array_equal(fd.age_cfp,d.age_cfp)
 F={k:np.load(fp/(k+'_SERVER_ONLY.npy'),mmap_mode='r') for k in ['z0','retina','mesh11']};F['demo']=d[['age_cfp','sex_cfp','bsa_cfp']].to_numpy(np.float32)
 assert F['mesh11'].shape==(len(d),11);raw=np.load(ROOT/'clinical_raw/observed_inputs_SERVER_ONLY.npy')
 dest=ROOT/'heads'/str(seed)/task/variant
 with stage(dest,seed=seed,task=task,variant=variant) as out:
  predictions={};thresholds={};selections=[];d0fold=[];d0final=None;start=clock.time()
  for arm,branches in list(ORDINARY.items())+list(RESIDUAL.items()):
   residual=arm in RESIDUAL;modeldir=out/arm;modeldir.mkdir();oof=np.full(len(d),np.nan);oof_risk=np.full((len(d),3),np.nan);epochs=[];fold_scores=[]
   for j,(tr,va) in enumerate(folds):
    offtr=predict(d0fold[j],F,tr,raw) if residual else None
    offva=predict(d0fold[j],F,va,raw) if residual else None
    b,p,hist=train(F,raw,tr,va,y,time,branches,pre,seed+j,harrell,offtr,offva)
    oof[va]=p;epochs.append(b['epoch']);fold_scores.append(max(h['selection_metric'] for h in hist))
    if not pre:
     rp=predict(b,F,tr,raw,offtr);b['baseline']=fit_baseline(rp,time[tr],y[tr]);oof_risk[va]=risks(p,b['baseline'])
    if residual:b['d0_bundle']=d0fold[j]
    if arm=='D0':d0fold.append(b)
    with (modeldir/f'fold{j}.pkl').open('xb') as f:pickle.dump(b,f)
    pd.DataFrame(hist).to_csv(modeldir/f'fold{j}_history.csv',index=False)
    save(out/'PROGRESS.json',dict(arm=arm,fold=j,selected_epoch=b['epoch'],seconds=clock.time()-start))
   final_epoch=int(np.median(epochs));off=predict(d0final,F,final,raw) if residual else None
   b,_,hist=train(F,raw,final,None,y,time,branches,pre,seed+100,harrell,off,None,final_epoch)
   alloff=predict(d0final,F,allidx,raw) if residual else None
   rp=predict(b,F,allidx,raw,alloff)
   assert np.isfinite(rp).all() and np.isfinite(oof[dev]).all()
   item={}
   if pre:
    predictions[arm+'__score']=expit(rp);predictions[arm+'__oof']=expit(oof)
    item['diagnosis']=threshold(y[dev],expit(oof[dev]))
   else:
    b['baseline']=fit_baseline(rp[final],time[final],y[final]);assert b['baseline']['max_followup']>=max(C['horizons'])
    predictions[arm+'__score']=rp;predictions[arm+'__oof']=oof;predictions[arm+'__risk']=risks(rp,b['baseline']);predictions[arm+'__oof_risk']=oof_risk
    for k,horizon in enumerate(C['horizons']):
     valid=(y[dev]&(time[dev]<=horizon))|(time[dev]>=horizon);ix=dev[valid];label=y[ix]&(time[ix]<=horizon)
     item[str(horizon)]=threshold(label,oof_risk[ix,k])
   if residual:b['d0_bundle']=d0final
   if arm=='D0':d0final=b
   with (modeldir/'final.pkl').open('xb') as f:pickle.dump(b,f)
   pd.DataFrame(hist).to_csv(modeldir/'final_history.csv',index=False)
   thresholds[arm]=item;selections.append(dict(arm=arm,branches=branches,input_dims=b['dims'],branch_projection=24,concatenated_dim=24*len(branches),offset=residual,fold_epochs=epochs,final_epoch=final_epoch,fold_primary=fold_scores,cv_mean=float(np.mean(fold_scores)),cv_sd=float(np.std(fold_scores,ddof=1)),train_n=len(final),train_positive_or_event=int(y[final].sum()),test_used_for_selection=False))
   save(out/'selection.json',selections);save(out/'thresholds.json',thresholds)
   np.savez(out/'predictions_PARTIAL_SERVER_ONLY.npz',**predictions)
   print(json.dumps(dict(seed=seed,task=task,variant=variant,arm=arm,final_epoch=final_epoch,seconds=clock.time()-start)),flush=True)
  # Fixed published clinical score, with fold-specific imputation for thresholds.
  clin=ClinicalScore().fit(raw[final]);score=clin.transform(raw)[:,0];oo=np.full(len(d),np.nan)
  for tr,va in folds:oo[va]=ClinicalScore().fit(raw[tr]).transform(raw[va])[:,0]
  predictions['ClinicalScore__score']=score;predictions['ClinicalScore__oof']=oo
  if pre:thresholds['ClinicalScore']={'diagnosis':threshold(y[dev],oo[dev])}
  else:
   predictions['ClinicalScore__risk']=np.repeat(score[:,None],3,axis=1);thresholds['ClinicalScore']={}
   for horizon in C['horizons']:
    valid=(y[dev]&(time[dev]<=horizon))|(time[dev]>=horizon);ix=dev[valid]
    thresholds['ClinicalScore'][str(horizon)]=threshold(y[ix]&(time[ix]<=horizon),oo[ix])
  with (out/'fixed_clinical_score.pkl').open('xb') as f:pickle.dump(clin,f)
  save(out/'thresholds.json',thresholds);np.savez(out/'predictions_SERVER_ONLY.npz',**predictions)
  save(out/'PROVENANCE.json',dict(seed=seed,task=task,variant=variant,manifest_sha256=sha(manifest(task)),feature_provenance=json.loads((fp/'PROVENANCE.json').read_text()),selection_rule='fixed-config fold early stopping; median selected epoch final refit',all_branches_24=True,fields11=FIELDS11,test_used_for_selection=False,cfp_age_anchor=True,student_frozen=True,formal_survival_coverage_verified=False,predictions_sha256=sha(out/'predictions_SERVER_ONLY.npz')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--task',choices=C['tasks'],required=True);p.add_argument('--variant',choices=C['variants'],required=True);a=p.parse_args();main(a.seed,a.task,a.variant)
