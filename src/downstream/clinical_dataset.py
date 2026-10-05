"""CFP-time disease feature dataset; unchanged sample construction.

Imports student modules only inside a configured private execution workspace.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from shared import S
from common import DEMO, VASC, PREPARED
from dataset import Cache


class Clinical(Dataset):
    def __init__(self,d,scopes=('paired','clinical')):
        self.d=d;self.cache=Cache(scopes);self.sc=dict(np.load(PREPARED/f'fold_{S["fold"]}'/'scalers.npz'))
    def __len__(self):return len(self.d)
    def __getitem__(self,i):
        row=self.d.iloc[i];xs=[];eyes=[];vals=[];masks=[]
        for side in ['L','R']:
            x,ok=self.cache.get(row[f'{side}_image_path']);xs.append(x);eyes.append(ok)
            raw=row[[f'{side}_{k}' for k in VASC]].to_numpy(float);mask=np.isfinite(raw)&ok
            vals.append(np.where(mask,(raw-self.sc['vascular_mean'])/self.sc['vascular_std'],0).astype('float32'));masks.append(mask)
        b={k:torch.from_numpy(np.stack([a[k] for a in xs])).float() for k in xs[0]};b['segment_mask']=b['segment_mask'].bool();b['images']=b.pop('tokens')
        b.update(eye_mask=torch.tensor(eyes,dtype=torch.bool),vessels=torch.tensor(np.stack(vals)),vessel_mask=torch.tensor(np.stack(masks)),demo_cfp=torch.tensor(row[DEMO].to_numpy('float32')),row_index=torch.tensor(i))
        return b
