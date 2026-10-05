from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data import MeshDataset, read_ed_exact
from losses import geometry, teacher_loss, sample_metrics, volume_curve
from model import make_model
from report_metrics import metrics as phenotype_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'seed': cfg['seed'], 'start': time.time()}
    try:
        preparation = json.loads((Path(cfg['data']['root']) / 'preparation/status.json').read_text())
        assert preparation['status'] == 'COMPLETED'
        assert hashlib.sha256(Path(cfg['data']['manifest']).read_bytes()).hexdigest() == preparation['derived_manifest_sha256']
        assert hashlib.sha256(Path(cfg['data']['topology']).read_bytes()).hexdigest() == preparation['topology_sha256']
        result['derived_manifest_sha256'] = preparation['derived_manifest_sha256']
        result['topology_sha256'] = preparation['topology_sha256']
        synthetic = np.arange(5, dtype=float)
        assert np.allclose(phenotype_metrics(synthetic, synthetic), [0, 0, 0, 1, 1, 1])
        assert np.allclose(phenotype_metrics(synthetic, synthetic + 1), [1, 1, 1, 1, 1, .5])
        assert np.isclose(phenotype_metrics(synthetic, synthetic + 4)[-1], -7.)
        result['metric_definition_tests'] = 'PASS: exact prediction, signed bias, and negative R2'
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        ds = MeshDataset(cfg['data']['manifest'], 'train')
        for idx in (0, 109, 1073, 2026, 8210, 23109, 37615, 46968):
            path = ds.table.iloc[idx].cache_path
            assert np.array_equal(read_ed_exact(path), np.load(path, allow_pickle=False)[:1])
        result['ed_exact_io_bitwise_match'] = True
        n = cfg['training']['evaluation_batch_per_gpu']
        batch = torch.stack([ds[j]['mesh'] for j in range(n)]).cuda()
        model = make_model(cfg).cuda()
        result['parameters'] = sum(p.numel() for p in model.parameters())
        result['counts'] = {s: len(MeshDataset(cfg['data']['manifest'], s)) for s in ('train', 'val', 'test')}
        assert result['counts'] == {'train': 46969, 'val': 2112, 'test': 4204}
        for q in ('LV', 'Myo'):
            for lev in range(1, 5):
                assert torch.allclose(model.topo.get(q, f'up_w{lev}').sum(1), torch.ones_like(model.topo.get(q, f'up_w{lev}')[:, 0]), atol=1e-6)
        ident, components = geometry(batch, batch, model.topo, cfg['loss'])
        result['identity_loss'] = float(ident.max())
        assert result['identity_loss'] < 1e-4, result
        model.set_stage('C')
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch, stage='C')
        loss, _ = teacher_loss(output, batch, model.topo, cfg['loss'])
        assert torch.isfinite(loss)
        loss.backward()
        grads = {name: float(p.grad.float().norm()) for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, missing
        assert all(math.isfinite(x) for x in grads.values())
        for group in ('shape_enc', 'motion_enc', 'shape_dec', 'motion_dec'):
            assert sum(x for name, x in grads.items() if name.startswith(group)) > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        result.update(one_batch_loss=float(loss), single_batch_seconds=time.time() - t0,
                      max_gpu_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                      missing_gradients=missing, finite_gradients=True)
        del output, loss, opt
        model.eval()
        with torch.no_grad():
            c, r = model.encode(batch[:1], 'C')
            zero = model.decode(c, torch.zeros_like(r), torch.tensor([0., .98, 1.], device='cuda'))
            result['zero_detail_difference_mm'] = float((zero['mesh_core_mm'] - zero['mesh_full_mm']).abs().max())
            result['cycle_position_error_mm'] = float((zero['mesh_core_mm'][:, 0] - zero['mesh_core_mm'][:, 2]).abs().max())
            assert result['zero_detail_difference_mm'] < 1e-5
            assert result['cycle_position_error_mm'] < 2e-3
            assert (zero['mesh_core_mm'][:, 0] - zero['mesh_core_mm'][:, 1]).abs().max() > 1e-7
            c2 = c.clone(); c2[:, 32:] += 1
            d2 = model.decode(c2, None, torch.tensor([0.], device='cuda'))
            assert torch.equal(zero['shape_core_mm'], d2['shape_core_mm'])
            omega = 2 * math.pi * torch.arange(1, model.k + 1, device='cuda')
            a, b = zero['A_core'], zero['B_core']
            deriv0 = (a * omega[None, :, None, None, None]).sum(1)
            deriv1 = (a * (omega * omega.cos())[None, :, None, None, None] - b * (omega * omega.sin())[None, :, None, None, None]).sum(1)
            result['cycle_derivative_error_mm_per_cycle'] = float((deriv0 - deriv1).abs().max())
            assert result['cycle_derivative_error_mm_per_cycle'] < 1e-2
            pred = model(batch[:1], stage='C')
            met = sample_metrics(pred, batch[:1], model.topo)
            assert all(torch.isfinite(v).all() for v in met.values())
            x = batch[0, :, 0].cpu().numpy().astype(np.float64)
            f = model.topo.LV_faces.cpu().numpy()
            expected = np.array([(xx[f[:, 0]] * np.cross(xx[f[:, 1]], xx[f[:, 2]])).sum() / 6000 for xx in x])
            actual = volume_curve(batch[:1, :, 0], model.topo.LV_faces).cpu().numpy()[0]
            result['volume_float64_max_abs_difference_ml'] = float(np.max(np.abs(expected - actual)))
            assert np.allclose(expected, actual, rtol=1e-5, atol=1e-3)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        c = c.detach().requires_grad_(True)
        decoded = model.decode(c)['mesh_core_mm']
        decoded.square().mean().backward()
        assert c.grad is not None and torch.isfinite(c.grad).all() and c.grad.norm() > 0
        result['frozen_decoder_student_gradient_norm'] = float(c.grad.norm())
        checkpoint = outdir / 'roundtrip.pt'
        torch.save(model.state_dict(), checkpoint)
        other = make_model(cfg).cuda()
        other.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        other.eval()
        with torch.no_grad():
            assert torch.allclose(other.decode(c.detach())['mesh_core_mm'], decoded.detach(), atol=1e-5, rtol=1e-5)
        result.update(status='SINGLE_GPU_PASS', file_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
                      config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(), elapsed_seconds=time.time() - result['start'])
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc))
        raise
    finally:
        (outdir / 'sanity.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
