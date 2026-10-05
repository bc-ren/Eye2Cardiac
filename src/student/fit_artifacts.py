"""Same prior/scaling/whitening estimators; no new held-out patient exposure."""
import numpy as np
import pandas as pd
import torch
from shared import *
from common import C,LATENT,VASC,DEMO,PREPARED,fit_prior

def artifacts(fold=0):
    assert fold==0
    p=PREP/S['fitted_artifacts_name'];audit=json.loads((p/'AUDIT.json').read_text())
    assert audit['status']=='COMPLETED'
    result=[]
    for name in ['prior_raw.npz','scalers.npz','residual_whitening.npz']:
        assert sha(p/name)==audit['hashes'][name]
        result.append(dict(np.load(p/name,allow_pickle=False)))
    return tuple(result)

def main():
    torch.set_num_threads(4)
    from model import DemographicPrior
    with stage(PREP/S['fitted_artifacts_name']) as out:
        # Async file membership is authoritative; legacy original_split is pre-fivefold metadata.
        # Same-day monitor is intentionally part of Train; only true holdouts block fitting.
        blocked=set().union(*(set(pd.read_parquet(PREP/f'{v}_{r}_SERVER_ONLY.parquet').eid.astype(str)) for v,r in [('same_day','test'),('asynchronous','validation'),('asynchronous','test')]))
        oldfit=PREPARED/'fold_0/prior_fit_SERVER_ONLY.parquet'
        safeids=set(pd.read_parquet(oldfit).eid.astype(str))-blocked
        root=Path(S['prior_source'])
        f=pd.read_parquet(root/'preparation/fit_manifest.parquet');z=pd.read_parquet(root/'latent_extraction/fit_latents_raw.parquet')
        f.eid=f.eid.astype(str);z.eid=z.eid.astype(str)
        f=f.merge(z[['eid',*LATENT]],on='eid',validate='one_to_one')
        safe=f[f.eid.isin(safeids)].copy();assert set(safe.eid)==safeids and safe.split.eq('train').all()
        raw=safe[LATENT].to_numpy(float);assert np.isfinite(raw).all()
        prior=fit_prior(safe[['age_cmr','sex','bsa_cmr']].to_numpy(float),raw,100.)
        np.savez(out/'prior_raw.npz',**prior)
        tr=pd.read_parquet(PREP/'joint_train_SERVER_ONLY.parquet');sync=pd.read_parquet(PREP/'same_day_train_SERVER_ONLY.parquet')
        v=np.concatenate([tr.loc[tr[s+'_image_path'].notna(),[s+'_'+k for k in VASC]].to_numpy(float) for s in ['L','R']])
        sc=dict(latent_mean=raw.mean(0),latent_std=np.maximum(raw.std(0),1e-6),latent_variance=raw.var(0),vascular_mean=np.nanmean(v,0),vascular_std=np.maximum(np.nanstd(v,0),1e-6))
        assert all(np.isfinite(a).all() for a in sc.values());np.savez(out/'scalers.npz',**sc)
        model=DemographicPrior(prior,sc).eval()
        with torch.no_grad():pred=model(torch.tensor(sync[DEMO].to_numpy('float32'))).numpy()
        target=(sync[LATENT].to_numpy(float)-sc['latent_mean'])/sc['latent_std'];res=target-pred
        center=res.mean(0);cov=np.cov(res,rowvar=False);alpha=C['whitening_shrinkage']
        eig,vec=np.linalg.eigh((1-alpha)*cov+alpha*np.eye(96)*np.trace(cov)/96)
        assert eig.min()>0 and np.isfinite(eig).all()
        matrix=(vec/np.sqrt(eig)[None])@vec.T;inverse=(vec*np.sqrt(eig)[None])@vec.T
        np.testing.assert_allclose(matrix@inverse,np.eye(96),atol=1e-7)
        np.savez(out/'residual_whitening.npz',matrix=matrix.astype('float32'),inverse=inverse.astype('float32'),center=center.astype('float32'),covariance=cov,eigenvalues=eig)
        safe[['eid','split','cmr_date']].to_parquet(out/'prior_fit_SERVER_ONLY.parquet',index=False)
        sync[['eid','retina_date','cmr_date','original_split']].to_parquet(out/'whitening_fit_SERVER_ONLY.parquet',index=False)
        tr[['eid','original_split','view']].to_parquet(out/'vascular_fit_SERVER_ONLY.parquet',index=False)
        assert blocked.isdisjoint(safeids) and blocked.isdisjoint(tr.eid.astype(str))
        save(out/'AUDIT.json',dict(status='COMPLETED',prior_fit_n=len(safe),prior_old_fit_n=len(pd.read_parquet(oldfit)),excluded_new_heldout_n=len(pd.read_parquet(oldfit))-len(safe),prior_alpha=100,whitening_n=len(sync),vascular_training_n=len(tr),heldout_overlap=0,old_prior_manifest_sha256=sha(oldfit),hashes={n:sha(out/n) for n in ['prior_raw.npz','scalers.npz','residual_whitening.npz']},test_used_for_fit=False))
        print((out/'AUDIT.json').read_text(),flush=True)
if __name__=='__main__':main()
