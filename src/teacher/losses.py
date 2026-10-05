"""Subject/surface-balanced losses in physical units and explicit scales."""
from __future__ import annotations

import torch


def huber(x, delta=1.):
    a = x.abs()
    return torch.where(a < delta, .5 * a.square() / delta, a - .5 * delta)


def volume_curve(x, faces):
    tri = x.float()[..., faces, :]
    signed = (tri[..., 0, :] * torch.cross(tri[..., 1, :], tri[..., 2, :], dim=-1)).sum(-1).sum(-1) / 6000.
    return signed


def lv_values(x, faces):
    signed = volume_curve(x[:, :, 0], faces)
    v = signed.abs()
    ed, es = v[:, 0], v.amin(1)
    return v, torch.stack([ed, es, 100 * (ed - es) / ed.clamp_min(1e-6)], -1), signed


def laplacian(x, idx, weights):
    return x - (x[..., idx, :] * weights[None, None, :, :, None]).sum(-2)


def geometry(pred, target, topo, cfg):
    pred, target = pred.float(), target.float()
    items = {}
    collect = {k: [] for k in ('vertex', 'edge', 'normal', 'lap_relative', 'excess_roughness',
                               'velocity', 'acceleration', 'face_area_floor')}
    for qi, q in enumerate(('LV', 'Myo')):
        p, t = pred[:, :, qi], target[:, :, qi]
        err = torch.linalg.vector_norm(p - t, dim=-1)
        collect['vertex'].append(huber(err).mean((1, 2)))
        e = topo.get(q, 'edges')
        lp = torch.linalg.vector_norm(p[:, :, e[:, 0]] - p[:, :, e[:, 1]], dim=-1)
        lt = torch.linalg.vector_norm(t[:, :, e[:, 0]] - t[:, :, e[:, 1]], dim=-1)
        collect['edge'].append((lp - lt).abs().mean((1, 2)))
        f = topo.get(q, 'faces')
        tp, tt = p[:, :, f], t[:, :, f]
        np_ = torch.cross(tp[..., 1, :] - tp[..., 0, :], tp[..., 2, :] - tp[..., 0, :], dim=-1)
        nt = torch.cross(tt[..., 1, :] - tt[..., 0, :], tt[..., 2, :] - tt[..., 0, :], dim=-1)
        ap, at = torch.linalg.vector_norm(np_, dim=-1), torch.linalg.vector_norm(nt, dim=-1)
        cosine = (np_ * nt).sum(-1) / (ap * at).clamp_min(1e-10)
        collect['normal'].append((1 - cosine.clamp(-1, 1)).mean((1, 2)))
        collect['face_area_floor'].append(torch.relu(.05 - ap / at.clamp_min(1e-8)).mean((1, 2)))
        idx, weights = topo.get(q, 'lap_indices'), topo.get(q, 'lap_weights')
        pl, tl = laplacian(p, idx, weights), laplacian(t, idx, weights)
        collect['lap_relative'].append(torch.linalg.vector_norm(pl - tl, dim=-1).mean((1, 2)))
        collect['excess_roughness'].append(torch.relu(torch.linalg.vector_norm(pl, dim=-1) -
                    1.1 * torch.linalg.vector_norm(tl, dim=-1) - .05).mean((1, 2)))
        vp, vt = p.roll(-1, 1) - p, t.roll(-1, 1) - t
        collect['velocity'].append(torch.linalg.vector_norm(vp - vt, dim=-1).mean((1, 2)))
        accp, acct = vp.roll(-1, 1) - vp, vt.roll(-1, 1) - vt
        collect['acceleration'].append(torch.linalg.vector_norm(accp - acct, dim=-1).mean((1, 2)))
    for k, values in collect.items():
        items[k] = torch.stack(values, 1).mean(1)
    vp, fp, _ = lv_values(pred, topo.LV_faces)
    vt, ft, _ = lv_values(target, topo.LV_faces)
    items['lv_volume'] = (vp - vt).abs().mean(1) / 10.
    items['lvef'] = huber((fp[:, 2] - ft[:, 2]) / 5.)
    items['ed_max'] = torch.relu(vp - vp[:, :1]).mean(1) / 10.
    pairp = pred[:, :, 1] - pred[:, :, 0]
    pairt = target[:, :, 1] - target[:, :, 0]
    items['lv_myo_pair'] = torch.linalg.vector_norm(pairp - pairt, dim=-1).mean((1, 2))
    total = sum(cfg['geom_weights'][k] * value for k, value in items.items())
    return total, items


def teacher_loss(output, target, topo, cfg):
    pc, pf = output['mesh_core_mm'].float(), output['mesh_full_mm'].float()
    target = target[:, :pc.shape[1]].float()
    lc, ic = geometry(pc, target, topo, cfg)
    lf, iff = geometry(pf, target, topo, cfg)
    keep = output['detail_keep'].float()
    full = (lf * keep).sum() / keep.sum().clamp_min(1)
    residual = (pf - pc).square().sum(-1).mean((1, 2, 3))
    residual = (residual * keep).sum() / keep.sum().clamp_min(1)
    jitter = pc.new_zeros(())
    if 'jitter_mesh_mm' in output:
        jitter = (output['jitter_mesh_mm'].float() - pc).square().sum(-1).mean()
    loss = full + cfg['core_weight'] * lc.mean() + cfg['residual_weight'] * residual + cfg['jitter_weight'] * jitter
    report = {'loss': loss.detach(), 'core_geometry': lc.mean().detach(), 'full_geometry': full.detach(),
              'residual_mm2': residual.detach(), 'jitter_mm2': jitter.detach()}
    report.update({'core_' + k: v.mean().detach() for k, v in ic.items()})
    report.update({'full_' + k: v.mean().detach() for k, v in iff.items()})
    return loss, report


@torch.no_grad()
def sample_metrics(output, target, topo):
    result = {}
    _, truth_lv, _ = lv_values(target, topo.LV_faces)
    es = volume_curve(target[:, :, 0], topo.LV_faces).argmin(1)
    for variant in ('core', 'full'):
        p = output[f'mesh_{variant}_mm'].float()
        if p.shape[1] == 1:
            tt = target[:, :1]
            es_here = torch.zeros_like(es)
        else:
            tt = target
            es_here = es
        dist = torch.linalg.vector_norm(p - tt, dim=-1)
        for qi, q in enumerate(('LV', 'Myo')):
            d = dist[:, :, qi]
            result[f'{variant}_{q}_mean_mm'] = d.mean((1, 2))
            result[f'{variant}_{q}_rmse_mm'] = d.square().mean((1, 2)).sqrt()
            result[f'{variant}_{q}_p95_mm'] = torch.quantile(d.flatten(1), .95, dim=1)
            result[f'{variant}_{q}_ed_rmse_mm'] = d[:, 0].square().mean(1).sqrt()
            result[f'{variant}_{q}_es_rmse_mm'] = d[torch.arange(len(d), device=d.device), es_here].square().mean(1).sqrt()
        volumes, vals, signed = lv_values(p, topo.LV_faces)
        result[f'{variant}_invalid_signed_volume_frames'] = (signed <= 0).sum(1).float()
        for j, phenotype in enumerate(('LVEDV', 'LVESV', 'LVEF')):
            result[f'{variant}_{phenotype}'] = vals[:, j]
        result[f'{variant}_es_frame'] = volumes.argmin(1).float()
    for j, phenotype in enumerate(('LVEDV', 'LVESV', 'LVEF')):
        result['reference_' + phenotype] = truth_lv[:, j]
    return result
