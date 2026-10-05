"""Original training-only demographic-filtered contrastive bank."""
import torch
from common import DEMO, LATENT

@torch.no_grad()
def bank_for(model,ds):
    device = next(model.parameters()).device
    d = ds.table
    demo = torch.tensor(d[DEMO].to_numpy('float32'),device=device)
    raw = d[LATENT].to_numpy('float32')
    target = torch.tensor((raw-ds.scalers['latent_mean'])/ds.scalers['latent_std'],device=device,dtype=torch.float32)
    return dict(demo=demo,white=model.whiten(target-model.prior(demo)).detach())

