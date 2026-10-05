"""Teacher-native cardiac fusion and time transport; active student inherits these modules."""
import torch
from torch import nn
from common import ROOT, load_module
base = load_module('query_reference_model', ROOT / 'reference/timebridge_model.py')
CardiacDecoder, DemographicPrior = base.CardiacDecoder, base.DemographicPrior
mlp, scale_gradient = base.mlp, base.scale_gradient

class CardiacQueryStudent(nn.Module):
    def __init__(self, c, prior, scalers, whiten, backbone=None):
        super().__init__()
        self.c = c
        if backbone is None:
            raise ValueError('Construct the published student.RetiCardiac with the explicit RetiZero backbone')
        self.backbone = backbone
        self.prior = DemographicPrior(prior, scalers).requires_grad_(False)
        self.register_buffer('white_matrix', torch.as_tensor(whiten['matrix'], dtype=torch.float32))
        self.register_buffer('white_center', torch.as_tensor(whiten['center'], dtype=torch.float32))
        self.register_buffer('shape_ids', torch.tensor(list(range(32)) + list(range(64, 80))))
        self.register_buffer('motion_ids', torch.tensor(list(range(32, 64)) + list(range(80, 96))))
        h, drop = c['hidden_dim'], c['dropout']
        self.query_project = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, h), nn.GELU())
        self.shared_seeds = nn.Parameter(torch.randn(1, 4, h) * .02)
        self.eye_embeddings = nn.Parameter(torch.zeros(1, 2, 1, h))
        self.cross_eye = nn.MultiheadAttention(h, 4, batch_first=True, dropout=0.)
        self.shared_norm = nn.LayerNorm(h)
        self.vascular = mlp(20, 64, h, drop)
        self.cardio_fusion = mlp(2*h, h, h, drop)
        self.shape_head = mlp(2*h, h, 48, drop)
        self.motion_head = mlp(2*h, h, 48, drop)
        for m in [self.shape_head, self.motion_head]:
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        self.vascular_aux = mlp(h, 64, 10)
        self.variance = mlp(3*h, 64, 96)
        nn.init.zeros_(self.variance[-1].weight); nn.init.zeros_(self.variance[-1].bias)
        self.transport = nn.Sequential(nn.Linear(100, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, c['transport_rank']), nn.Tanh(), nn.Linear(c['transport_rank'], 96, bias=False))
        nn.init.zeros_(self.transport[-1].weight)

    def set_lora(self, enabled):
        for name, p in self.backbone.named_parameters():
            if any('.' + k + '.' in name for k in ['aq', 'av', 'bq', 'bv']):
                p.requires_grad_(enabled)

    def whiten(self, residual):
        return (residual.float() - self.white_center) @ self.white_matrix

    def forward(self, images, eye_mask, vessels, vessel_mask, demo_cfp, backbone_gradient=1.):
        b = len(images)
        assert images.shape[1:3] == (2, 3) and eye_mask.shape == (b, 2)
        assert eye_mask.dtype == torch.bool and eye_mask.any(1).all()
        assert vessels.shape == vessel_mask.shape == (b, 2, 10)
        valid = eye_mask.flatten()
        cls, query = self.backbone(images.flatten(0, 1)[valid])
        raw = cls.new_zeros((2*b, 1024)); raw[valid] = cls
        qe = query.new_zeros((2*b, 4, 1024)); qe[valid] = query
        raw, qe = raw.reshape(b, 2, 1024), qe.reshape(b, 2, 4, 1024)
        qe = self.query_project(qe) + self.eye_embeddings
        keys = qe.flatten(1, 2)
        shared, att = self.cross_eye(self.shared_seeds.expand(b, -1, -1), keys, keys,
            key_padding_mask=(~eye_mask).repeat_interleave(4, 1), need_weights=True)
        shared = self.shared_norm(shared + self.shared_seeds)
        ew = eye_mask.float() / eye_mask.sum(1, keepdim=True)
        v = self.vascular(torch.cat([torch.where(vessel_mask, vessels, 0), vessel_mask.float()], -1))
        v = (v * ew[..., None]).sum(1)
        cardiac = self.cardio_fusion(torch.cat([shared[:, :3], v[:, None].expand(-1, 3, -1)], -1))
        # Semantic roles are guided by the native teacher shape/motion targets;
        # a separate myocardium latent is not claimed.
        shape = self.shape_head(torch.cat([cardiac[:, 0], cardiac[:, 1]], -1))
        motion = self.motion_head(torch.cat([cardiac[:, 2], cardiac[:, 0]], -1))
        r0 = torch.cat([shape[:, :32], motion[:, :32], shape[:, 32:], motion[:, 32:]], 1).float()
        # Scale the ENTIRE baseline branch gradient for async loss, not just ViT.
        r0 = scale_gradient(r0, backbone_gradient)
        prior0 = self.prior(demo_cfp)
        var = .05 + 3.95 * torch.sigmoid(self.variance(cardiac.flatten(1)).float())
        return dict(r0=r0, z0=prior0+r0, prior0=prior0, demo0=demo_cfp,
            retina=(raw*ew[..., None]).sum(1).float(), shared_tokens=shared.float(),
            eye_specific=(qe-shared[:, None]).float(), attention=att.float(),
            vascular_prediction=self.vascular_aux(shared[:, 3]).float(),
            variance0=scale_gradient(var,backbone_gradient))

    def propagate(self, out, delta_years):
        dt = delta_years.float().reshape(-1, 1)
        assert torch.isfinite(dt).all() and (dt >= 0).all()
        gate = dt / (self.c['time_scale_years'] + dt)
        time = torch.cat([dt/self.c['time_scale_years'], torch.log1p(dt), gate,
                          (out['demo0'][:, :1]-60)/10], 1)
        change = gate * self.transport(torch.cat([out['r0'], time], 1)).float()
        demo1 = out['demo0'].clone(); demo1[:, 0] += dt[:, 0]
        prior1 = self.prior(demo1)
        return dict(z1=prior1+out['r0']+change, residual1=out['r0']+change,
                    prior1=prior1, change=change)

    def export(self):
        # Include all adapters even while LoRA is frozen in warm-up.
        return {k: v.detach().cpu() for k,v in self.state_dict().items()
                if not k.startswith('backbone.') or k == 'backbone.queries'
                or any('.'+s+'.' in k for s in ['aq','av','bq','bv'])}

    def restore(self, state):
        assert set(state) == set(self.export()), 'Incomplete adapter/preprocessing state'
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys and all(k.startswith('backbone.') for k in result.missing_keys)
