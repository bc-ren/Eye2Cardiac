"""Memory-only execution change: preserve logical-batch KD and loss reductions."""
import torch
from torch.utils.checkpoint import checkpoint
from shared import S


def vector_chunks(original, z, b, decoder, geometry, c):
    losses, statistics = [], []
    # All these statistics are participant-wise in the original implementation.
    names = ['mesh_mean_mm', 'selection_score', 'invalid_fraction', 'EF_MAE_pp']
    for start in range(0, len(z), S['physical_micro_batch']):
        stop = min(start + S['physical_micro_batch'], len(z))
        target = b['mesh_mm'][start:stop]
        def calculate(zz, mm):
            ll, ss, _, _ = original(zz, {'mesh_mm': mm}, decoder, geometry, c)
            return ll, torch.stack([ss[k] for k in names], 1)
        if torch.is_grad_enabled():
            ll, ss = checkpoint(calculate, z[start:stop], target, use_reentrant=False)
        else:
            ll, ss = calculate(z[start:stop], target)
        losses.append(ll)
        statistics.append(ss)
    ss = torch.cat(statistics)
    return torch.cat(losses), {k: ss[:, j] for j, k in enumerate(names)}, None, None


def scalar_chunks(original, z, b, decoder, geometry, c):
    names = ['mesh_mean_mm', 'mesh_mse_mm2', 'ED_mean_mm', 'ES_mean_mm', 'LV_mean_mm',
             'Myo_mean_mm', 'EF_MAE_pp', 'invalid_fraction', 'relative_curve_mae',
             'es_neighborhood_huber', 'selection_score']
    losses, stats, counts = [], [], []
    for start in range(0, len(z), S['physical_micro_batch']):
        stop = min(start + S['physical_micro_batch'], len(z))
        def calculate(zz, mm):
            ll, ss, _, _ = original(zz, {'mesh_mm': mm}, decoder, geometry, c)
            return ll, torch.stack([ss[k] for k in names])
        if torch.is_grad_enabled():
            ll, ss = checkpoint(calculate, z[start:stop], b['mesh_mm'][start:stop], use_reentrant=False)
        else:
            ll, ss = calculate(z[start:stop], b['mesh_mm'][start:stop])
        losses.append(ll * (stop - start))
        stats.append(ss * (stop - start))
        counts.append(stop - start)
    ss = torch.stack(stats).sum(0) / sum(counts)
    return torch.stack(losses).sum() / sum(counts), {k: ss[j] for j, k in enumerate(names)}, None, None
