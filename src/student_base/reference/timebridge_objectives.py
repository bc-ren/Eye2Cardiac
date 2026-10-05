"""Exact five-component physical training descriptor (not the Mesh11 report)."""
import torch

def describe(mesh, decoder):
    assert mesh.ndim == 5 and mesh.shape[1:] == (50, 2, 6141, 3)
    curves = []
    for qi, name in enumerate(['LV', 'Myo']):
        x = mesh[:, :, qi].float()
        x = x - x.mean(2, keepdim=True)
        faces = getattr(decoder.teacher.topo, name + '_faces').long()
        tri = x[:, :, faces]
        vol = (tri[..., 0, :] * torch.linalg.cross(tri[..., 1, :], tri[..., 2, :], dim=-1)).sum((-1, -2)) / 6000
        curves.append(vol)
    curves = torch.stack(curves, -1)
    lv, outer = curves.unbind(-1)
    ed, ed_index = lv.max(1); es, es_index = lv.min(1)
    row = torch.arange(len(mesh), device=mesh.device)
    mass = 1.05 * (outer[row, ed_index] - ed)
    ef = 100 * (ed - es) / ed.clamp_min(1e-3)
    invalid = (lv <= 0).any(1) | (outer <= lv).any(1) | (mass <= 0) | (ef < 0) | (ef > 100)
    return dict(curves=curves, ph=torch.stack([ed, es, ed - es, ef, mass], 1),
                ed=ed_index, es=es_index, invalid=invalid)

