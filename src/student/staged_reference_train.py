"""Fixed-seed staged two-GPU training; explicit apparent/held-out selection."""
import argparse,copy,shutil,signal,time
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset
from shared import *
from common import C,PREPARED
from dataset import CachedPaired,Cache,global_batches,ShardBatches
from student import make_models,args_of
from data import worker_init,device_batch
from engine import bank_for
from temporal_objective import loss_for,async_vector,temporal_weights
from objectives import loss_for as original_loss_for,relational
from vector_reference import physical_vector,relational_vector

def cfg():return dict(C,repair_variant='head_init',workers=S['workers'],cpu_threads=S['cpu_threads'])

def optimizer(m, c, epoch=0):
    # Exact original AdamW recipe, now including the requested trainable retina.
    assert S['retina_learning_rate'] == S['fixed_learning_rate']
    return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
        lr=S['fixed_learning_rate']*S['lr_decay_per_epoch']**epoch,
        weight_decay=c['weight_decay'])


def loader(ds,c,indices=None,seed=0):
    w=c.get('workers',S['workers'])
    return DataLoader(ds if indices is None else Subset(ds,list(map(int,indices))),batch_size=c.get('batch_size',4),
        num_workers=w,pin_memory=True,persistent_workers=w>0,worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(seed))

def combine_validation(sync,asynchronous):
    assert sync['n']>0 and asynchronous['n']>0
    return dict(selection_score=.5*sync['selection_score']+.5*asynchronous['selection_score'],
        invalid_fraction=max(sync['invalid_fraction'],asynchronous['invalid_fraction']),
        same_day=sync,asynchronous=asynchronous,weights=[.5,.5],
        same_day_monitor_is_apparent=True,independent_cross_domain_validation=False)

@torch.no_grad()
def evaluate(m,dec,geo,ds,c,batch,rank=0,world=1,max_batches=None):
    m.eval();dec.eval();tot=torch.zeros(5,device='cuda',dtype=torch.float64)
    for i,b in enumerate(loader(ds,dict(c,batch_size=batch),range(rank,len(ds),world))):
        if max_batches is not None and i>=max_batches:break
        b=device_batch(b,'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):o=m(**args_of(b))
        z=m.propagate(o,b['delta_years'])['z1']
        _,s,_,_=physical_vector(z,b,dec,geo,c)
        tot+=torch.stack([torch.tensor(len(z),device='cuda'),s['selection_score'].sum(),s['invalid_fraction'].sum(),s['mesh_mean_mm'].sum(),s['EF_MAE_pp'].sum()]).double()
    if world>1:dist.all_reduce(tot)
    v=tot.cpu().tolist();assert v[0]>0
    return dict(n=int(v[0]),selection_score=v[1]/v[0],invalid_fraction=v[2]/v[0],mesh_mean_mm=v[3]/v[0],EF_MAE_pp=v[4]/v[0])

def require_finite_grads(m,require_scan=False):
    grads={k:p.grad for k,p in m.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    if require_scan:
        for i in [20,21,22,23]:
            prefix=f'backbone.vit.blocks.{i}.'
            assert any(k.startswith(prefix) and g.abs().sum()>0 for k,g in grads.items()),prefix
        from retizero_backbone import assert_scope
        assert_scope(m.backbone)
        assert all(p.grad is None for p in m.backbone.vit.parameters() if not p.requires_grad)
        for part in ['patch_encoder.adapter.core.scans','patch_encoder.adapter.polar.radial','patch_encoder.adapter.polar.angular']:
            assert any(k.startswith(part) and g.abs().sum()>0 for k,g in grads.items()),part

