"""One mixed-batch forward, unchanged per-domain losses, sample-count reduction."""
import pandas as pd
import torch
from torch import nn
from shared import PREP
from student import args_of
from model import scale_gradient
from temporal_objective import loss_for


def bank_lookup(joint, same_day):
    index = {str(eid): i for i, eid in enumerate(same_day.eid)}
    result = [index.get(str(eid), -1) for eid in joint.eid]
    sync = joint['view'].eq('same_day').to_numpy()
    assert all((value >= 0) == bool(is_sync) for value, is_sync in zip(result, sync))
    return torch.tensor(result, dtype=torch.long)


def subset(values, mask):
    n = len(mask)
    return {k: (v[mask] if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == n else v)
            for k, v in values.items()}


class JointSystem(nn.Module):
    def __init__(self, m, decoder, geometry, c, bank):
        super().__init__()
        self.student, self.decoder, self.geometry = m, decoder, geometry
        self.c, self.bank = c, bank
        joint = pd.read_parquet(PREP/'same_day_train_SERVER_ONLY.parquet', columns=['eid', 'view'])
        sync = pd.read_parquet(PREP/'same_day_train_SERVER_ONLY.parquet', columns=['eid'])
        self.register_buffer('sync_bank_index', bank_lookup(joint, sync).to(next(m.parameters()).device))

    def train(self, mode=True):
        super().train(mode)
        self.decoder.eval()
        self.student.prior.eval()
        return self

    def loss_from_output(self, out, b):
        asynchronous = b['delta_years'] > 0
        assert torch.equal(asynchronous, b['is_async'])
        assert not asynchronous.any(), 'Asynchronous student training is not authorized'
        # Retain the old asynchronous shared-branch Jacobian multiplier, not
        # a sampling ratio. Forward values are exactly unchanged.
        factor = torch.where(asynchronous, self.c['async_shared_gradient'], 1.).float()[:, None]
        out = dict(out)
        out['r0'] = scale_gradient(out['r0'], factor)
        out['variance0'] = scale_gradient(out['variance0'], factor)
        out['z0'] = out['prior0'] + out['r0']
        loss = out['r0'].sum()*0
        for is_async in [False, True]:
            mask = asynchronous if is_async else ~asynchronous
            n = int(mask.sum())
            if not n:
                continue
            bb = subset(b, mask)
            # Explicit patient-ID mapping; never confuse joint row indices
            # with the same-day contrastive bank's row order.
            if not is_async:
                bb['row_index'] = self.sync_bank_index[bb['row_index']]
                assert (bb['row_index'] >= 0).all()
            oo = {k: out[k][mask] for k in ['r0', 'z0', 'prior0', 'demo0', 'variance0', 'vascular_prediction']}
            group_loss = loss_for(self.student, oo, bb, self.decoder, self.geometry, self.c,
                                  bank=self.bank if not is_async else None,
                                  asynchronous=is_async, anchor=None)[0]
            # Natural mixture: each subgroup contributes its actual N/B.
            # No equal-domain rebalancing or division by temporal-weight sum.
            loss = loss + group_loss * (n / len(asynchronous))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite mixed student loss')
        return loss

    def forward(self, b):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.student(**args_of(b))
        return self.loss_from_output(out, b)
