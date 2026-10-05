"""Explicit frozen-prior marginalization; never impute observed patient sex."""
import torch
from torch import nn

class MissingSexPrior(nn.Module):
    def __init__(self, frozen):
        super().__init__()
        self.frozen = frozen
        self.male_fraction = float(frozen.mean[1])
        assert 0 < self.male_fraction < 1

    def forward(self, demo):
        assert torch.isfinite(demo[:, 0]).all()
        missing = ~torch.isfinite(demo[:, 1])
        if not missing.any():
            return self.frozen(demo)
        female = demo.clone()
        female[missing, 1] = 0
        result = self.frozen(female)
        male = demo[missing].clone()
        male[:, 1] = 1
        result[missing] = ((1-self.male_fraction)*result[missing]
                           + self.male_fraction*self.frozen(male))
        return result

def verify(frozen):
    adapter = MissingSexPrior(frozen)
    x = torch.tensor([[45., 0., 1.7], [70., 1., float('nan')],
                      [65., float('nan'), 2.1]], device=frozen.mean.device)
    before = x.clone()
    a = adapter(x)
    torch.testing.assert_close(a[:2], frozen(x[:2]), rtol=0, atol=0)
    f, m = x[2:].clone(), x[2:].clone()
    f[:, 1] = 0; m[:, 1] = 1
    expected = (1-adapter.male_fraction)*frozen(f)+adapter.male_fraction*frozen(m)
    # GEMM accumulation may differ at ~1e-7 between batch sizes 1 and 3.
    torch.testing.assert_close(a[2:], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x, before, equal_nan=True, rtol=0, atol=0)
    assert torch.isfinite(a).all()
    return adapter, dict(known_sex_exact=True, missing_sex_manual_average_max_error=float((a[2:]-expected).abs().max()),
                         input_not_mutated=True, male_fraction=adapter.male_fraction)
