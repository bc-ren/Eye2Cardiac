"""Exact prior/decoder primitives used by the 35-epoch student."""
import torch
from torch import nn
from common import load_module

def mlp(ni, h, no, dropout=0.):
    return nn.Sequential(nn.Linear(ni, h), nn.LayerNorm(h), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(h, no))


def scale_gradient(x, scale):
    """Same forward value, exact scaled backward Jacobian."""
    return x.detach() + scale * (x - x.detach())


class DemographicPrior(nn.Module):
    """Frozen raw-latent ridge converted to the new safe-fit latent scale."""
    def __init__(self, prior, scalers):
        super().__init__()
        for k in ['median', 'mean', 'std', 'coef']:
            self.register_buffer(k, torch.as_tensor(prior[k], dtype=torch.float32))
        for k in ['latent_mean', 'latent_std']:
            self.register_buffer(k, torch.as_tensor(scalers[k], dtype=torch.float32))

    def forward(self, demo):
        x = demo.float()
        assert x.ndim == 2 and x.shape[1] == 3 and torch.isfinite(x[:, :2]).all()
        assert ((x[:, 1] == 0) | (x[:, 1] == 1)).all()
        m = ~torch.isfinite(x[:, 2])
        a, s = (x[:, 0] - 60) / 10, x[:, 1]
        b = (torch.where(m, self.median, x[:, 2]) - 2) / .3
        f = torch.stack([a, s, b, a*a, a*s, b*s, a*b, m.float()], 1)
        with torch.autocast(x.device.type, enabled=False):
            f = torch.cat([torch.ones_like(a[:, None]), (f - self.mean) / self.std], 1)
            return (f @ self.coef - self.latent_mean) / self.latent_std


class CardiacDecoder(nn.Module):
    def __init__(self, c, scalers):
        super().__init__()
        mod = load_module('timebridge_teacher', c['teacher_code'])
        ck = torch.load(c['teacher_checkpoint'], map_location='cpu', weights_only=True)
        assert ck['epoch'] == 84 and ck['stage'] == 'C'
        cfg = ck['cfg']; cfg['data']['topology'] = c['topology']
        self.teacher = mod.make_model(cfg)
        self.teacher.load_state_dict(ck['model'], strict=True)
        self.teacher.requires_grad_(False)
        for k in ['latent_mean', 'latent_std']:
            self.register_buffer(k, torch.tensor(scalers[k], dtype=torch.float32))
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, z):
        with torch.autocast(z.device.type, enabled=False):
            raw = z.float() * self.latent_std + self.latent_mean
            return self.teacher.decode(raw[:, :64], raw[:, 64:])['mesh_full_mm']

