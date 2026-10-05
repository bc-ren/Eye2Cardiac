"""Original caches and exact original asynchronous membership; explicit age anchor."""
import importlib.util
import pandas as pd
import numpy as np
import torch
from shared import ROOT,PREP,S
spec=importlib.util.spec_from_file_location('immutable_parent_dataset',ROOT/'reference/dataset.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from raw_dataset import RawCache
class Cache(RawCache):
    pass

class CachedPaired(base.CachedPaired):
    def __init__(self,split,view,cache=None):
        name='joint_train' if view=='joint' else view+'_'+split
        self.table=pd.read_parquet(PREP/(name+'_SERVER_ONLY.parquet')).reset_index(drop=True)
        self.table['age_cfp']=self.table.age_model
        self.scalers=dict(np.load(PREP/S['fitted_artifacts_name']/'scalers.npz',allow_pickle=False))
        self.cache=cache or Cache();self.view=view
    def __getitem__(self,i):
        b=super().__getitem__(i)
        b['is_async']=torch.tensor(self.table.iloc[i]['view']=='asynchronous')
        return b

def global_batches(n,batch,seed,epoch):
    ix=np.random.default_rng(seed+epoch).permutation(n)
    batches=[ix[i:i+2*batch] for i in range(0,n,2*batch)]
    if len(batches[-1])==1 and len(batches)>1:
        batches[-2]=np.r_[batches[-2],batches[-1]];batches.pop()
    assert np.array_equal(np.sort(np.concatenate(batches)),np.arange(n))
    return batches

class ShardBatches:
    def __init__(self,batches,rank):self.items=[b[rank::2].tolist() for b in batches];assert all(self.items)
    def __iter__(self):return iter(self.items)
    def __len__(self):return len(self.items)
