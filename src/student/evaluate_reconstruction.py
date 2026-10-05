"""Locked Test: physical 50 frames and Mesh11; original evaluation semantics."""
import argparse,importlib.util,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd,torch
from shared import *
from staged_reference_train import cfg,loader
from dataset import CachedPaired,Cache
from data import device_batch
from student import make_models,args_of
from objectives import describe
FIELDS=['lv_edv_ml','lv_esv_ml','lv_sv_ml','lv_ef_pct','lv_mass_g','lv_mean_wall_thickness_ed_mm','lv_wall_thickness_p95_ed_mm','lv_global_wall_thickening_pct','lv_base_apex_length_ed_mm','lv_base_apex_length_es_mm','lv_longitudinal_shortening_pct']
WRAPPER=ROOT/'geometry_bridge.py'
W=None
def init_geometry():
    global W
    s=importlib.util.spec_from_file_location('physical13_worker',WRAPPER);W=importlib.util.module_from_spec(s);s.loader.exec_module(W);W.worker_init()
    os.sched_setaffinity(0,set(range(os.cpu_count())))
def geometry_job(payload):return W.phenotype_job(payload)

def metrics(x,y):
    vx,vy=x.var(),y.var();cov=((x-x.mean())*(y-y.mean())).mean();diff=x-y
    return dict(MAE=np.abs(diff).mean(),RMSE=np.sqrt(np.mean(diff**2)),Pearson=cov/np.sqrt(vx*vy) if vx*vy>0 else np.nan,
        R2=1-np.mean(diff**2)/vx if vx>0 else np.nan,CCC=2*cov/(vx+vy+(x.mean()-y.mean())**2) if vx+vy>0 else np.nan,
        predicted_SD=np.sqrt(vy),target_SD=np.sqrt(vx))

def report(frame,out):
    groups={'all':np.ones(len(frame),bool)}
    for k in frame.status_at_cfp.unique():groups[str(k)]=frame.status_at_cfp.eq(k).to_numpy()
    for k in frame.diagnosis_time_relation.unique():groups['time_'+str(k)]=frame.diagnosis_time_relation.eq(k).to_numpy()
    rows=[];rng=np.random.default_rng(S['seed'])
    for group,mask in groups.items():
        q=frame[mask]
        for name in FIELDS:
            x=q['target_'+name].to_numpy(float);y=q['pred_'+name].to_numpy(float)
            if len(q)<3:
                rows.append(dict(group=group,phenotype=name,n=len(q),status='INSUFFICIENT_N'));continue
            assert np.isfinite(x).all() and np.isfinite(y).all()
            point=metrics(x,y);boots={k:[] for k in point}
            for _ in range(S['bootstrap_replicates']):
                ix=rng.integers(len(q),size=len(q));v=metrics(x[ix],y[ix])
                for k in boots:boots[k].append(v[k])
            for k,v in point.items():
                bb=np.asarray(boots[k]);bb=bb[np.isfinite(bb)];lo,hi=np.quantile(bb,[.025,.975]) if len(bb) else [np.nan,np.nan]
                rows.append(dict(group=group,phenotype=name,n=len(q),metric=k,value=v,ci_low=lo,ci_high=hi,valid_bootstrap=len(bb),status='ESTIMATED'))
    pd.DataFrame(rows).to_csv(out/'mesh11_metrics_CI.csv',index=False)
    errors=[c for c in frame if c.startswith('error_')]
    pd.DataFrame([dict(group=g,n=int(m.sum()),**{k:float(frame.loc[m,k].mean()) if m.any() else None for k in errors}) for g,m in groups.items()]).to_csv(out/'physical_mesh_error_mm.csv',index=False)

def main(run,phase,view,smoke):
    torch.set_num_threads(2);c=cfg();c['batch_size']=4;c['workers']=2
    path=ROOT/'runs'/run/(phase+'_best.pt');assert status(path.parent)=='COMPLETED'
    if not smoke:assert (ROOT/'selection/student_SELECTION.json').exists(),'Select hyperparameters before Test'
    ck=torch.load(path,map_location='cpu',weights_only=False);c=relocated_config(ck['config'],c);c.update(batch_size=4,workers=2);m,dec,geo=make_models(c,ck['arm']);m.restore(ck['model']);m.eval()
    ds=CachedPaired('validation' if smoke else 'test',view,Cache())
    # Prepared asynchronous Test is the full original Test, including interval events.
    indices=range(min(4,len(ds))) if smoke else None
    dest=ROOT/('reconstruction_smoke' if smoke else 'reconstruction')/run/(phase+'_'+view)
    with stage(dest,run=run,phase=phase,view=view,smoke=smoke) as out,ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_geometry) as pool,torch.inference_mode():
        rows=[]
        for b in loader(ds,c,indices):
            ix=b['row_index'].numpy();b=device_batch(b,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):o=m(**args_of(b))
            z=m.propagate(o,b['delta_years'])['z1'] if view=='asynchronous' else o['z0']
            pred=dec(z);target=b['mesh_mm'];p,t=describe(pred,dec),describe(target,dec)
            assert torch.isfinite(pred).all();rr=torch.arange(len(z),device='cuda')
            pp=torch.stack([pred[rr,p['ed']],pred[rr,p['es']]],1).cpu().numpy();tt=torch.stack([target[rr,t['ed']],target[rr,t['es']]],1).cpu().numpy()
            jobs=[(int(idx),str(ds.table.iloc[idx].eid),pp[j],int(p['ed'][j]),int(p['es'][j])) for j,idx in enumerate(ix)]
            gtjobs=[(int(idx),str(ds.table.iloc[idx].eid),tt[j],int(t['ed'][j]),int(t['es'][j])) for j,idx in enumerate(ix)]
            results=list(pool.map(geometry_job,jobs));truth=list(pool.map(geometry_job,gtjobs))
            err=torch.linalg.vector_norm(pred-target,dim=-1);rms=err.square().mean((1,2,3)).sqrt();means=err.mean((1,2,3));ed=err[rr,t['ed']].mean((1,2));es=err[rr,t['es']].mean((1,2))
            for j,idx in enumerate(ix):
                row=ds.table.iloc[idx];cfp=pd.Timestamp(row.retina_date);date=pd.to_datetime(row.get('first_exact_record_date'),errors='coerce')
                status_='known_CHD_at_CFP' if pd.notna(date) and date<=cfp else 'no_prior_recorded_CHD'
                if bool(row.get('unknown_exact_date',False)) and status_!='known_CHD_at_CFP':status_='unresolved_time_evidence'
                cmr=pd.Timestamp(row.cmr_date)
                relation=('on_or_before_CFP' if date<=cfp else 'between_CFP_CMR' if date<=cmr else 'after_CMR') if pd.notna(date) else 'no_exact_record_date'
                r=dict(row_id=int(idx),eid=str(row.eid),cohort='UKB-Dev',view=view,status_at_cfp=status_,diagnosis_time_relation=relation,delta_years=float(row.delta_years),error_mean_mm=float(means[j]),error_RMSE_mm=float(rms[j]),error_ED_mm=float(ed[j]),error_ES_mm=float(es[j]),error_LV_mm=float(err[j,:,0].mean()),error_Myo_mm=float(err[j,:,1].mean()),invalid_geometry=bool(p['invalid'][j]))
                for field in FIELDS:r['pred_'+field]=results[j][field];r['target_'+field]=truth[j][field]
                rows.append(r)
            save(out/'PROGRESS.json',dict(done=len(rows),total=4 if smoke else len(ds)))
        frame=pd.DataFrame(rows);frame.to_parquet(out/'patient_predictions_SERVER_ONLY.parquet',index=False)
        assert np.isfinite(frame[['pred_'+f for f in FIELDS]+['target_'+f for f in FIELDS]].to_numpy(float)).all(),'Invalid geometry: retain and audit, no silent exclusions'
        report(frame,out)
        save(out/'PROVENANCE.json',dict(checkpoint=path,checkpoint_sha256=sha(path),code=fingerprints(),geometry_code_sha256=sha(WRAPPER),
            physical_units='mm, mL, g, %, dimensionless ratios',affine_used=False,target='original teacher-training physical target mesh',
            async_semantics='Known-interval transported estimate at CMR time; NOT proven CFP-time truth' if view=='asynchronous' else 'same-day cardiac transfer',
            n=len(rows),invalid_mesh_n=int(frame.invalid_geometry.sum()),test_used_for_selection=False,bootstrap_conditional_on_fitted_model=True,
            full_locked_asynchronous_test_retained=not smoke and view=='asynchronous',interval_events_not_excluded_from_test=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase',choices=['sync_only'],required=True);p.add_argument('--view',choices=['same_day','asynchronous'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.run,a.phase,a.view,a.smoke)
