"""CardiacAE: LV/myocardium autoencoder with deterministic core64/detail32."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

MODEL_REGISTRY = {}


def register_model(name):
    def decorate(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f'Duplicate model: {name}')
        MODEL_REGISTRY[name] = cls
        return cls
    return decorate


def make_model(cfg):
    return MODEL_REGISTRY[cfg['architecture_id']](cfg)


class Topology(nn.Module):
    def __init__(self, path):
        super().__init__()
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                self.register_buffer(key, torch.from_numpy(z[key].copy()))

    def get(self, q, key):
        return getattr(self, q + '_' + key)


class SpiralBlock(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.fc = nn.Linear(9 * ni, no)
        self.norm = nn.LayerNorm(no)
        self.skip = nn.Linear(ni, no, bias=False)

    def forward(self, x, spiral):
        neighbors = x[..., spiral, :].flatten(-2)
        return torch.nn.functional.gelu(self.norm(self.fc(neighbors)) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, ni, channels, use_checkpoint):
        super().__init__()
        self.blocks = nn.ModuleList(SpiralBlock(a, b) for a, b in zip([ni] + channels[:-1], channels))
        self.use_checkpoint = use_checkpoint

    def forward(self, x, topo, q):
        for level, block in enumerate(self.blocks):
            ids = topo.get(q, f'spiral{level}')
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, ids, use_reentrant=False)
            else:
                x = block(x, ids)
            x = x[..., topo.get(q, f'down{level + 1}'), :]
        return x


class Pool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x):
        return (self.score(x).softmax(dim=-2) * x).sum(-2)


class SharedFusion(nn.Module):
    def __init__(self, dim, heads, layers):
        super().__init__()
        self.attn_a = nn.ModuleList(nn.MultiheadAttention(dim, heads, dropout=.1, batch_first=True) for _ in range(layers))
        self.attn_b = nn.ModuleList(nn.MultiheadAttention(dim, heads, dropout=.1, batch_first=True) for _ in range(layers))
        self.norm_a = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.norm_b = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.ff_a = nn.ModuleList(nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Dropout(.1), nn.Linear(1024, dim)) for _ in range(layers))
        self.ff_b = nn.ModuleList(nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Dropout(.1), nn.Linear(1024, dim)) for _ in range(layers))
        self.final_a = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.final_b = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.pool = Pool(dim)
        self.out = nn.Linear(dim, 8)

    def forward(self, a, b):
        for aa, ab, na, nb, fa, fb, la, lb in zip(self.attn_a, self.attn_b, self.norm_a, self.norm_b,
                                               self.ff_a, self.ff_b, self.final_a, self.final_b):
            da = aa(a, b, b, need_weights=False)[0]
            db = ab(b, a, a, need_weights=False)[0]
            a, b = na(a + da), nb(b + db)
            a, b = la(a + fa(a)), lb(b + fb(b))
        return self.out(self.pool(torch.cat([a, b], dim=1)))


class GraphDecoder(nn.Module):
    def __init__(self, ni, channels, no):
        super().__init__()
        self.channels = channels
        self.seed = nn.Linear(ni, 24 * channels[0])
        self.blocks = nn.ModuleList(SpiralBlock(a, b) for a, b in zip(channels, channels[1:] + [channels[-1]]))
        self.output = nn.Linear(channels[-1], no)
        nn.init.normal_(self.output.weight, std=1e-4)
        nn.init.zeros_(self.output.bias)

    def forward(self, code, topo, q):
        h = self.seed(code).reshape(len(code), 24, self.channels[0])
        for level, block in zip((4, 3, 2, 1), self.blocks):
            idx, w = topo.get(q, f'up_idx{level}'), topo.get(q, f'up_w{level}')
            h = (h[:, idx, :] * w[None, :, :, None]).sum(-2)
            h = block(h, topo.get(q, f'spiral{level - 1}'))
        return self.output(h)


@register_model('CardiacAE')
@register_model('lv_myo_nonode_core64_detail32_v1')
class CardiacAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.topo = Topology(cfg['data']['topology'])
        m = cfg['model']
        self.k = m['harmonics']
        ch = m['encoder_channels']
        dim = ch[-1]
        self.shape_enc = nn.ModuleDict({q: Encoder(3, ch, m['activation_checkpoint']) for q in ('LV', 'Myo')})
        self.motion_enc = nn.ModuleDict({q: Encoder(6, ch, m['activation_checkpoint']) for q in ('LV', 'Myo')})
        self.shape_pool = nn.ModuleDict({q: Pool(dim) for q in ('LV', 'Myo')})
        self.motion_pool = nn.ModuleDict({q: Pool(dim) for q in ('LV', 'Myo')})
        self.shape_private = nn.ModuleDict({q: nn.Linear(dim, 20) for q in ('LV', 'Myo')})
        self.motion_private = nn.ModuleDict({q: nn.Linear(dim, 20) for q in ('LV', 'Myo')})
        self.shape_shared = SharedFusion(dim, 8, 2)
        self.motion_shared = SharedFusion(dim, 8, 2)
        self.temporal = nn.ModuleDict()
        self.phase_embed = nn.ModuleDict()
        self.time_pool = nn.ModuleDict()
        for q in ('LV', 'Myo'):
            layer = nn.TransformerEncoderLayer(dim, 8, 1024, .1, activation='gelu', batch_first=True, norm_first=True)
            self.temporal[q] = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
            self.phase_embed[q] = nn.Linear(2, dim)
            self.time_pool[q] = Pool(dim)
        dc, dr = m['core_decoder_channels'], m['detail_decoder_channels']
        self.shape_dec = nn.ModuleDict({q: GraphDecoder(20, dc, 3) for q in ('LV', 'Myo')})
        self.motion_dec = nn.ModuleDict({q: GraphDecoder(40, dc, 6 * self.k) for q in ('LV', 'Myo')})
        self.shape_detail = nn.ModuleDict({q: GraphDecoder(28, dr, 3) for q in ('LV', 'Myo')})
        self.motion_detail = nn.ModuleDict({q: GraphDecoder(56, dr, 6 * self.k) for q in ('LV', 'Myo')})
        self.register_buffer('jitter_std', torch.zeros(64))

    def set_stage(self, stage):
        for name, p in self.named_parameters():
            motion = name.startswith(('motion_', 'temporal.', 'phase_embed.', 'time_pool.'))
            p.requires_grad_(stage == 'C' or (stage == 'A' and not motion) or (stage == 'B' and motion))

    def encode(self, mesh_mm, stage='C'):
        valid_frames = (1, 50) if stage == 'A' else (50,)
        if mesh_mm.ndim != 5 or mesh_mm.shape[1] not in valid_frames or mesh_mm.shape[2:] != (2, 6141, 3):
            raise ValueError(f'Expected Bx50x2x6141x3, got {mesh_mm.shape}')
        x = mesh_mm / 100.
        shape_tokens, motion_tokens = {}, {}
        for qi, q in enumerate(('LV', 'Myo')):
            ed = x[:, 0, qi] - self.topo.template_mm[qi] / 100.
            shape_tokens[q] = self.shape_enc[q](ed, self.topo, q)
            if stage != 'A':
                disp = x[:, :, qi] - x[:, :1, qi]
                velocity = x[:, :, qi].roll(-1, 1) - x[:, :, qi]
                h = self.motion_enc[q](torch.cat([disp, velocity], -1), self.topo, q)
                b, t, v, d = h.shape
                h = h.permute(0, 2, 1, 3).reshape(b * v, t, d)
                phase = torch.arange(t, device=x.device, dtype=x.dtype) / 50.
                pe = torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], -1)
                h = h + self.phase_embed[q](pe)[None]
                if self.cfg['model']['activation_checkpoint'] and self.training and torch.is_grad_enabled():
                    h = checkpoint(self.temporal[q], h, use_reentrant=False)
                else:
                    h = self.temporal[q](h)
                motion_tokens[q] = self.time_pool[q](h).reshape(b, v, d)
        s = {q: self.shape_private[q](self.shape_pool[q](shape_tokens[q])) for q in shape_tokens}
        cs = torch.cat([s['LV'][:, :12], s['Myo'][:, :12], self.shape_shared(shape_tokens['LV'], shape_tokens['Myo'])], -1)
        rs = torch.cat([s['LV'][:, 12:], s['Myo'][:, 12:]], -1)
        if stage == 'A':
            cm, rm = torch.zeros_like(cs), torch.zeros_like(rs)
        else:
            m = {q: self.motion_private[q](self.motion_pool[q](motion_tokens[q])) for q in motion_tokens}
            cm = torch.cat([m['LV'][:, :12], m['Myo'][:, :12], self.motion_shared(motion_tokens['LV'], motion_tokens['Myo'])], -1)
            rm = torch.cat([m['LV'][:, 12:], m['Myo'][:, 12:]], -1)
        return torch.cat([cs, cm], -1), torch.cat([rs, rm], -1)

    def decode(self, core, detail=None, phase=None, stage='C'):
        if core.ndim != 2 or core.shape[1] != 64:
            raise ValueError('core must have width 64')
        if detail is not None and detail.shape != (len(core), 32):
            raise ValueError('detail must have width 32')
        phase = torch.arange(50, device=core.device, dtype=core.dtype) / 50 if phase is None else phase
        shape_core, shape_full, a_core, b_core, a_full, b_full = [], [], [], [], [], []
        for qi, q in enumerate(('LV', 'Myo')):
            s = torch.cat([core[:, qi * 12:qi * 12 + 12], core[:, 24:32]], -1)
            m = torch.cat([core[:, 32 + qi * 12:44 + qi * 12], core[:, 56:64]], -1)
            shape = self.topo.template_mm[qi] + 100 * self.shape_dec[q](s, self.topo, q)
            coeff = core.new_zeros(len(core), 6141, 6 * self.k) if stage == 'A' else 100 * self.motion_dec[q](torch.cat([s, m], -1), self.topo, q)
            sf, cf = shape, coeff
            if detail is not None:
                rs = detail[:, qi * 8:qi * 8 + 8]
                rm = detail[:, 16 + qi * 8:24 + qi * 8]
                s_in = torch.cat([s, rs], -1)
                s_zero = torch.cat([s, torch.zeros_like(rs)], -1)
                sf = shape + 100 * (self.shape_detail[q](s_in, self.topo, q) - self.shape_detail[q](s_zero, self.topo, q))
                if stage != 'A':
                    m_in = torch.cat([s, m, rs, rm], -1)
                    m_zero = torch.cat([s, m, torch.zeros_like(rs), torch.zeros_like(rm)], -1)
                    cf = coeff + 100 * (self.motion_detail[q](m_in, self.topo, q) - self.motion_detail[q](m_zero, self.topo, q))
            cc = coeff.reshape(len(core), 6141, 2, self.k, 3)
            ff = cf.reshape(len(core), 6141, 2, self.k, 3)
            shape_core.append(shape); shape_full.append(sf)
            a_core.append(cc[:, :, 0].transpose(1, 2)); b_core.append(cc[:, :, 1].transpose(1, 2))
            a_full.append(ff[:, :, 0].transpose(1, 2)); b_full.append(ff[:, :, 1].transpose(1, 2))
        sc, sf = torch.stack(shape_core, 1), torch.stack(shape_full, 1)
        ac, bc = torch.stack(a_core, 2), torch.stack(b_core, 2)
        af, bf = torch.stack(a_full, 2), torch.stack(b_full, 2)
        angles = 2 * math.pi * phase.float()[:, None] * torch.arange(1, self.k + 1, device=core.device)[None]
        def compose(s, a, b):
            return s.float()[:, None] + torch.einsum('tk,bkqvc->btqvc', angles.sin(), a.float()) + torch.einsum('tk,bkqvc->btqvc', angles.cos() - 1, b.float())
        return {'mesh_core_mm': compose(sc, ac, bc), 'mesh_full_mm': compose(sf, af, bf),
                'shape_core_mm': sc, 'shape_full_mm': sf, 'A_core': ac, 'B_core': bc,
                'A_full': af, 'B_full': bf}

    def forward(self, mesh_mm, stage='C', enable_jitter=False):
        core, detail = self.encode(mesh_mm, stage)
        phase = torch.zeros(1, device=mesh_mm.device) if stage == 'A' else None
        keep = torch.ones(len(core), device=core.device, dtype=torch.bool)
        if self.training:
            keep = torch.rand(len(core), device=core.device) >= self.cfg['model']['detail_dropout_probability']
        out = self.decode(core, detail * keep[:, None], phase, stage)
        out.update(core64=core, detail32=detail, detail_keep=keep, labels=mesh_mm, logits=out['mesh_full_mm'])
        if enable_jitter:
            noisy = core + torch.randn_like(core) * self.jitter_std * .02
            out['jitter_mesh_mm'] = self.decode(noisy, None, phase, stage)['mesh_core_mm']
        return out


# Preserve historical imports and checkpoint architecture IDs without changing
# any registered parameter/buffer names or the numerical model implementation.
LVMyoTeacher = CardiacAE
