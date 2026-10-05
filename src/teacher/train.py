"""DDP training with immutable code snapshots and explicit run state."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data import MeshDataset, DistributedEvalSampler
from losses import teacher_loss, sample_metrics
from model import make_model


def json_save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset, cfg, rank, world, training, batch_size=None):
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=cfg['seed'], drop_last=False) if training else DistributedEvalSampler(dataset, rank, world)
    generator = torch.Generator().manual_seed(cfg['seed'] + rank)
    return DataLoader(dataset, batch_size=batch_size or cfg['training']['evaluation_batch_per_gpu'], sampler=sampler,
                      num_workers=cfg['training']['workers_per_gpu'], pin_memory=True,
                      persistent_workers=cfg['training']['workers_per_gpu'] > 0,
                      worker_init_fn=worker_seed, generator=generator, drop_last=False,
                      **({'prefetch_factor': 2} if cfg['training']['workers_per_gpu'] > 0 else {}))


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(model, data, stage, rank, world):
    model.eval()
    records = []
    for batch in data:
        mesh = batch['mesh'].cuda(non_blocking=True)
        with autocast():
            out = model(mesh, stage=stage)
        metrics = sample_metrics(out, mesh, model.topo)
        host = {k: v.float().cpu().numpy() for k, v in metrics.items()}
        for j, eid in enumerate(batch['eid']):
            records.append({'eid': eid, **{k: float(v[j]) for k, v in host.items()}})
    all_records = [None] * world
    dist.all_gather_object(all_records, records)
    records = [r for part in all_records for r in part]
    if len(set(r['eid'] for r in records)) != len(records):
        raise RuntimeError('Repeated validation/test subject')
    frame = pd.DataFrame(records).sort_values('eid')
    means = frame.drop(columns='eid').mean().to_dict()
    rc = .5 * (means['core_LV_rmse_mm'] + means['core_Myo_rmse_mm'])
    rf = .5 * (means['full_LV_rmse_mm'] + means['full_Myo_rmse_mm'])
    pc = .5 * (means['core_LV_p95_mm'] + means['core_Myo_p95_mm'])
    means.update(J_geom=rc + .5 * rf + .1 * pc, n=len(frame))
    if not all(math.isfinite(float(x)) for x in means.values()):
        raise FloatingPointError('Non-finite evaluation metric')
    return frame, means


@torch.no_grad()
def latent_statistics(model, dataset, cfg, rank, world, output):
    model.eval()
    dl = loader(dataset, cfg, rank, world, False)
    accum = torch.zeros(129, dtype=torch.float64, device='cuda')
    for batch in dl:
        with autocast():
            core, _ = model.encode(batch['mesh'].cuda(non_blocking=True), 'C')
        core = core.double()
        accum[0] += len(core)
        accum[1:65] += core.sum(0)
        accum[65:] += core.square().sum(0)
    dist.all_reduce(accum)
    if int(accum[0]) != len(dataset):
        raise RuntimeError('Latent statistics participant count mismatch')
    mean = accum[1:65] / accum[0]
    std = ((accum[65:] / accum[0]) - mean.square()).clamp_min(0).sqrt()
    model.jitter_std.copy_(std.float())
    if rank == 0:
        np.savez(output, mean=mean.cpu().numpy(), std=std.cpu().numpy(), n=int(accum[0]), source='train_only')
    del dl


def set_train_mode(model, stage):
    model.train()
    if stage == 'B':
        for name, module in model.named_children():
            if name.startswith('shape_'):
                module.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', dest='run', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--sanity-record', required=True)
    parser.add_argument('--max-steps', type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed != cfg['seed']:
        raise ValueError('Explicit seed and config differ')
    sanity = json.loads(Path(args.sanity_record).read_text())
    if sanity['status'] != 'PASS':
        raise ValueError('Sanity has not passed')
    # A formal job must run the exact code/config that passed preflight.
    for filename, expected in sanity['file_hashes'].items():
        if hashlib.sha256((Path(__file__).parent / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Code changed after preflight: {filename}')
    if hashlib.sha256(Path(args.config).read_bytes()).hexdigest() != sanity['config_sha256']:
        raise ValueError('Config changed after preflight')
    if hashlib.sha256(Path(cfg['data']['manifest']).read_bytes()).hexdigest() != sanity['derived_manifest_sha256']:
        raise ValueError('Derived manifest changed after preflight')
    if hashlib.sha256(Path(cfg['data']['topology']).read_bytes()).hexdigest() != sanity['topology_sha256']:
        raise ValueError('Topology changed after preflight')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run = Path(args.run)
    if rank == 0:
        run.mkdir(parents=True, exist_ok=False)
        for sub in ('checkpoints', 'metrics', 'source'):
            (run / sub).mkdir()
        for p in Path(__file__).parent.glob('*.py'):
            shutil.copy2(p, run / 'source' / p.name)
        shutil.copy2(args.config, run / 'config.yaml')
        shutil.copy2(args.sanity_record, run / 'sanity.json')
        json_save(run / 'provenance.json', {
            'argv': sys.argv, 'seed': args.seed, 'world_size': world,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE', 'TORCH_NCCL_ASYNC_ERROR_HANDLING')},
            'parent_checkpoint': None, 'initialization': 'random_from_scratch',
            'code_hashes': sanity['file_hashes'], 'git_commit': None,
            'git_state': 'source workspace is not a Git repository; exact source archived and hashed',
            'nondeterministic': ['CUDA scatter/backward kernels may be nondeterministic', 'TF32 enabled', 'BF16 autocast'],
            'training_sampler': 'seeded DistributedSampler; padding documented per epoch',
            'evaluation_sampler': 'strided without padding; EID uniqueness asserted',
            'test_selection': False, 'clinical_wall_thickness': 'not anatomically validated; disabled',
            'full_intersection_QA': 'pending separate geometric evaluator; not claimed by training metrics'})
    dist.barrier()
    start = time.time()
    status = {'status': 'RUNNING', 'start_time': start, 'epoch': 0, 'stage': 'A'}
    if rank == 0:
        json_save(run / 'status.json', status)
    try:
        model = make_model(cfg).cuda()
        seed_all(cfg['seed'] + rank)
        train_ds = MeshDataset(cfg['data']['manifest'], 'train')
        val_ds = MeshDataset(cfg['data']['manifest'], 'val')
        test_ds = MeshDataset(cfg['data']['manifest'], 'test')
        if (len(train_ds), len(val_ds), len(test_ds)) != (46969, 2112, 4204):
            raise ValueError('Derived split counts changed')
        val_dl = loader(val_ds, cfg, rank, world, False)
        best = float('inf')
        global_epoch, global_step = 0, 0
        optimizer_steps = 0
        for stage_cfg in cfg['training']['stages']:
            stage = stage_cfg['stage']
            stage_ds = MeshDataset(cfg['data']['manifest'], 'train', ed_only=stage == 'A')
            bs = stage_cfg['microbatch_per_gpu']
            accum = stage_cfg['gradient_accumulation']
            train_dl = loader(stage_ds, cfg, rank, world, True, batch_size=bs)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.set_stage(stage)
            ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=stage_cfg['lr'], weight_decay=cfg['training']['weight_decay'])
            for se in range(1, stage_cfg['epochs'] + 1):
                global_epoch += 1
                train_dl.sampler.set_epoch(global_epoch)
                set_train_mode(model, stage)
                lrbase = stage_cfg['lr']
                warm = cfg['training']['warmup_epochs']
                factor = se / warm if se <= warm else .5 * (1 + math.cos(math.pi * (se - warm) / max(1, stage_cfg['epochs'] - warm)))
                lr = cfg['training']['min_lr'] + (lrbase - cfg['training']['min_lr']) * factor
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                epoch_start = time.time()
                n_steps = len(train_dl)
                running = torch.zeros(2, dtype=torch.float64, device='cuda')
                for step, batch in enumerate(train_dl):
                    mesh = batch['mesh'].cuda(non_blocking=True)
                    final_micro = (step + 1) % accum == 0 or step + 1 == n_steps
                    wstart = (step // accum) * accum
                    window_total = sum(min(bs, len(train_dl.sampler) - j * bs) for j in range(wstart, min(wstart + accum, n_steps)))
                    jitter = stage == 'C' and se > 5 and global_step % 4 == 0
                    with (contextlib.nullcontext() if final_micro else ddp.no_sync()):
                        with autocast():
                            out = ddp(mesh, stage=stage, enable_jitter=jitter)
                        loss, report = teacher_loss(out, mesh, model.topo, cfg['loss'])
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Nonfinite loss epoch={global_epoch} step={step}')
                        (loss * len(mesh) / window_total).backward()
                    if final_micro:
                        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['training']['gradient_clip_norm'], error_if_nonfinite=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                    running[0] += loss.detach().double() * len(mesh)
                    running[1] += len(mesh)
                    global_step += 1
                    if step % 25 == 0 or step + 1 == n_steps:
                        free, total = torch.cuda.mem_get_info()
                        memory = {'rank': rank, 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                                  'reserved_GiB': torch.cuda.memory_reserved() / 2**30,
                                  'used_fraction': (total-free) / total}
                        gpu_memory = [None] * world
                        dist.all_gather_object(gpu_memory, memory)
                    if rank == 0 and (step % 25 == 0 or step + 1 == n_steps):
                        record = {'stage': stage, 'epoch': global_epoch, 'stage_epoch': se,
                                  'microstep': step + 1, 'microsteps_in_epoch': n_steps,
                                  'optimizer_steps': optimizer_steps, 'lr': lr,
                                  'loss': float(loss.detach()), 'elapsed_epoch_s': time.time() - epoch_start,
                                  'microbatch_per_gpu': bs, 'effective_batch_size': bs * world * accum,
                                  'gpu_memory': gpu_memory,
                                  'max_gpu_allocated_gb': torch.cuda.max_memory_allocated() / 1e9}
                        print(json.dumps(record), flush=True)
                        status.update(record, status='RUNNING', total_elapsed_s=time.time() - start)
                        json_save(run / 'status.json', status)
                        with (run / 'metrics/train_steps.jsonl').open('a') as f:
                            f.write(json.dumps(record) + '\n')
                    del out, loss, mesh
                    if args.max_steps and global_step >= args.max_steps:
                        break
                dist.all_reduce(running)
                frame, vm = evaluate(model, val_dl, stage, rank, world)
                if len(frame) != len(val_ds):
                    raise RuntimeError('Validation count differs')
                if rank == 0:
                    cp = run / 'checkpoints' / f'epoch_{global_epoch:03d}_{stage}.pt'
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'cfg': cfg,
                                'epoch': global_epoch, 'stage': stage, 'validation': vm,
                                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}, cp)
                    record = {'stage': stage, 'epoch': global_epoch, 'train_loss': float(running[0] / running[1]),
                              'train_sampler_n': int(running[1]), 'train_unique_n': len(train_ds),
                              'sampler_padding_n': int(running[1]) - len(train_ds),
                              'epoch_seconds': time.time() - epoch_start, 'validation': vm}
                    with (run / 'metrics/epochs.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                    frame.to_csv(run / 'metrics' / f'validation_epoch_{global_epoch:03d}.csv', index=False)
                    if stage == 'C' and vm['J_geom'] < best:
                        best = vm['J_geom']
                        json_save(run / 'geometry_best.json', {'checkpoint': str(cp), 'epoch': global_epoch, 'J_geom': best})
                    print(json.dumps(record), flush=True)
                dist.barrier()
                if args.max_steps and global_step >= args.max_steps:
                    break
            del ddp, optimizer, train_dl, stage_ds
            if stage == 'B' and not args.max_steps:
                latent_statistics(model, train_ds, cfg, rank, world, run / 'stageB_train_latent_statistics.npz')
            if args.max_steps:
                break
        if not args.max_steps:
            best_info = json.loads((run / 'geometry_best.json').read_text())
            checkpoint_data = torch.load(best_info['checkpoint'], map_location='cpu')
            model.load_state_dict(checkpoint_data['model'])
            del checkpoint_data
            test_dl = loader(test_ds, cfg, rank, world, False)
            frame, tm = evaluate(model, test_dl, 'C', rank, world)
            if len(frame) != 4204:
                raise RuntimeError('Test count mismatch')
            latent_statistics(model, train_ds, cfg, rank, world, run / 'final_train_latent_statistics.npz')
            if rank == 0:
                frame.to_csv(run / 'metrics/test_geometry_best_per_subject.csv', index=False)
                json_save(run / 'metrics/test_geometry_best_summary.json', tm)
                status.update(status='REPORTING', epoch=global_epoch)
                json_save(run / 'status.json', status)
                from report_metrics import generate_report
                generate_report(run / 'metrics/test_geometry_best_per_subject.csv', run / 'metrics/bootstrap_reports')
                json_save(run / 'transfer_best.json', {'status': 'PENDING', 'reason': 'Fixed paired student probe is a separate follow-up; no transfer-optimal claim'})
            dist.barrier()
        if rank == 0:
            status.update(status='COMPLETED', total_elapsed_s=time.time() - start,
                          geometry_training_complete=True, transfer_probe_status='PENDING',
                          full_intersection_QA_status='PENDING')
            json_save(run / 'status.json', status)
    except BaseException as exc:
        failure = {'status': 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                   'error': repr(exc), 'traceback': traceback.format_exc(), 'rank': rank, 'time': time.time()}
        json_save(run / f'failure_rank{rank}.json', failure)
        if rank == 0:
            status.update(failure)
            json_save(run / 'status.json', status)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
