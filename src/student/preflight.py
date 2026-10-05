"""Validate relocated input contracts and numerical equivalence before training.

Creates no training run. Refuses to authorize training without the GPU checks.
The apparent same-day monitor overlap is explicitly checked, not concealed.
"""
import argparse
import json
import numpy as np
import pandas as pd
from shared import ROOT, PREP, S, sha, save, stage, fingerprints, seed_all


def validate_inputs():
    from common import C
    expected = json.loads((ROOT/'INPUT_HASHES.json').read_text())
    for file, digest in expected.items():
        if sha(file) != digest:
            raise ValueError('Input changed after configuration: '+file)
    assert sha(S['retina_checkpoint']) == S['retina_sha256']
    assert sha(C['teacher_checkpoint']) == C['teacher_sha256']
    counts = {'same_day_train':1777, 'same_day_validation':355, 'same_day_test':200,
              'asynchronous_train':7172, 'asynchronous_validation':1793, 'asynchronous_test':2341,
              'joint_train':8949}
    tables = {}
    for name, n in counts.items():
        d = pd.read_parquet(PREP/(name+'_SERVER_ONLY.parquet'))
        assert len(d)==n and d.eid.is_unique, name
        assert np.isfinite(d.age_model).all(), name
        assert 'age_cfp' in d and np.isfinite(d.age_cfp).all(), name
        np.testing.assert_allclose(d.age_model,d.age_cfp,rtol=0,atol=1e-6,
                                   err_msg='age_model must equal original CFP-anchored age, not CMR age')
        dt=(pd.to_datetime(d.cmr_date)-pd.to_datetime(d.retina_date)).dt.days.to_numpy()/365.25
        np.testing.assert_allclose(d.delta_years,dt,atol=.01,rtol=0)
        if name.startswith('same_day'): assert (dt==0).all()
        if name.startswith('asynchronous'): assert (dt>0).all()
        tables[name]=d
    ids=lambda n:set(tables[n].eid.astype(str))
    assert len(ids('same_day_train') & ids('same_day_validation'))==355
    holdouts=['same_day_test','asynchronous_validation','asynchronous_test']
    for n in holdouts:
        assert ids('same_day_train').isdisjoint(ids(n)), n
    for i,n in enumerate(holdouts):
        for other in holdouts[i+1:]: assert ids(n).isdisjoint(ids(other))
    manifest=pd.read_parquet(PREP/'image_manifest_SERVER_ONLY.parquet')
    assert manifest.image_path.is_unique and manifest.token_id.is_unique
    needed=set()
    for d in tables.values():
        for side in ['L','R']: needed.update(d[side+'_image_path'].dropna().astype(str))
    assert needed.issubset(set(manifest.image_path.astype(str)))
    from pathlib import Path
    for row in manifest[manifest.image_path.isin(needed)].itertuples():
        p=Path(row.preprocessed_rgb if row.spatial_complete else row.image_path)
        assert p.is_file() and p.stat().st_size>0, 'Missing image; no silent sample skipping'
    from fit_artifacts import artifacts
    artifacts()  # verifies immutable fitted-artifact hashes
    return dict(counts=counts, age_anchor='CFP', apparent_monitor_overlap=355,
                test_used_for_selection=False, changed_input_hashes=0, code=fingerprints())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu-check',action='store_true')
    args=parser.parse_args()
    with stage(ROOT/'preflight') as out:
        result=validate_inputs()
        result['gpu_checked']=False
        if args.gpu_check:
            import torch
            from dataset import Cache,CachedPaired
            from staged_reference_train import cfg,loader,require_finite_grads
            from student import make_models,args_of
            from data import device_batch
            from objectives import physical,_physical_full
            from vector_reference import physical_vector,_physical_vector_full
            torch.set_num_threads(4);seed_all(S['seed'])
            c=cfg();ds=CachedPaired('train','same_day',Cache())
            b=device_batch(next(iter(loader(ds,dict(c,batch_size=5,workers=0),range(5)))),'cuda')
            m,dec,geo=make_models(c);m.eval();dec.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): o=m(**args_of(b))
            assert o['z0'].shape==(5,96) and torch.isfinite(o['z0']).all()
            parity={}
            for name,full,chunk in [('scalar',_physical_full,physical),('vector',_physical_vector_full,physical_vector)]:
                z=o['z0'].detach().float().requires_grad_(True)
                a=full(z,b,dec,geo,c)[0].mean();ga=torch.autograd.grad(a,z)[0]
                z2=z.detach().clone().requires_grad_(True)
                v=chunk(z2,b,dec,geo,c)[0].mean();gb=torch.autograd.grad(v,z2)[0]
                torch.testing.assert_close(a,v,rtol=3e-5,atol=3e-5)
                torch.testing.assert_close(ga,gb,rtol=3e-4,atol=3e-5)
                parity[name]=dict(loss_difference=float((a-v).abs()),gradient_max_difference=float((ga-gb).abs().max()))
            m.train();m.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): o=m(**args_of(b))
            o['z0'].square().mean().backward();require_finite_grads(m,True)
            result.update(gpu_checked=True,physical_chunk_equivalence=parity)
        save(out/'AUDIT.json',result)
    if args.gpu_check:
        save(ROOT/'PREFLIGHT_COMPLETED.json',dict(status='COMPLETED',code=fingerprints(),audit=result))
    else:
        print('Input checks passed. Training remains locked until --gpu-check passes in a fresh configured run.')


if __name__=='__main__':main()
