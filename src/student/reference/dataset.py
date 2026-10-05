"""Exact active paired-sample transform; membership is supplied by dataset.py."""
import numpy as np, pandas as pd, torch
from common import DEMO, LATENT, VASC
from shared import S


class CachedPaired(torch.utils.data.Dataset):
    def __init__(self,split,view,cache=None):
        raise RuntimeError('Use dataset.CachedPaired with explicitly prepared manifests')
    def __len__(self):return len(self.table)
    def __getitem__(self,i):
        row=self.table.iloc[i];items=[];eyes=[];vals=[];masks=[]
        for side in ['L','R']:
            item,ok=self.cache.get(row[f'{side}_image_path']);items.append(item);eyes.append(ok)
            raw=row[[f'{side}_{k}' for k in VASC]].to_numpy(float);mask=np.isfinite(raw)&ok
            vals.append(np.where(mask,(raw-self.scalers['vascular_mean'])/self.scalers['vascular_std'],0).astype('float32'));masks.append(mask)
        z=(row[LATENT].to_numpy(float)-self.scalers['latent_mean'])/self.scalers['latent_std']
        mesh=np.load(row.cache_path);assert mesh.shape==(50,2,6141,3) and np.isfinite(mesh).all()
        b={k:torch.from_numpy(np.stack([a[k] for a in items])).float() for k in items[0]}
        b['segment_mask']=b['segment_mask'].bool();b['images']=b.pop('tokens')
        b.update(eye_mask=torch.tensor(eyes,dtype=torch.bool),vessels=torch.tensor(np.stack(vals)),vessel_mask=torch.tensor(np.stack(masks)),
            demo_cfp=torch.tensor(row[DEMO].to_numpy('float32')),latent_std=torch.tensor(z,dtype=torch.float32),delta_years=torch.tensor(row.delta_years,dtype=torch.float32),
            mesh_mm=torch.from_numpy(mesh),row_index=torch.tensor(i))
        return b

