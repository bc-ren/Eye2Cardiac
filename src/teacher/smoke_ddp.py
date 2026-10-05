from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from data import MeshDataset
from losses import teacher_loss
from model import make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--sanity', required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    torch.manual_seed(cfg['seed'])
    model = make_model(cfg).cuda()
    model.jitter_std.fill_(.1)
    torch.manual_seed(cfg['seed'] + rank)
    history = []
    started = time.time()
    try:
        for sc in cfg['training']['stages']:
            stage = sc['stage']; n = sc['microbatch_per_gpu']
            ds = MeshDataset(cfg['data']['manifest'], 'train', ed_only=stage == 'A')
            batch = torch.stack([ds[rank * n + j]['mesh'] for j in range(n)]).cuda()
            model.set_stage(stage); model.train()
            if stage == 'B':
                for name, module in model.named_children():
                    if name.startswith('shape_'): module.eval()
            ddp = DDP(model, device_ids=[rank], broadcast_buffers=False)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
            torch.cuda.reset_peak_memory_stats()
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = ddp(batch, stage=stage, enable_jitter=stage == 'C' and step == 2)
                loss, _ = teacher_loss(out, batch, model.topo, cfg['loss'])
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                optimizer.step(); torch.cuda.synchronize()
                free, total = torch.cuda.mem_get_info()
                row = {'rank': rank, 'stage': stage, 'batch': n, 'step': step, 'loss': float(loss), 'gradient_norm': float(norm),
                       'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
                       'reserved_GiB': torch.cuda.memory_reserved() / 2**30, 'device_used_fraction': (total-free)/total}
                history.append(row); print(json.dumps(row), flush=True)
                if step == 2:
                    assert row['device_used_fraction'] >= cfg['training']['target_training_device_memory_fraction_minimum'], row
                del out, loss
            del ddp, optimizer, batch, ds
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            dist.barrier()
        # Checks that parameters remain numerically identical after all-reduced updates.
        checksum = torch.stack([p.detach().float().sum() for p in model.parameters()])
        reference = checksum.clone()
        dist.broadcast(reference, 0)
        difference = (reference - checksum).abs().max()
        assert difference < 1e-5, difference
        all_history = [None] * dist.get_world_size()
        dist.all_gather_object(all_history, history)
        if rank == 0:
            result = json.loads(Path(args.sanity).read_text())
            assert result['status'] == 'SINGLE_GPU_PASS'
            result.update(status='PASS', ddp_gpus=dist.get_world_size(), ddp_steps_per_stage=3,
                          nccl_transport_environment={k: os.environ.get(k) for k in ('NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE', 'TORCH_NCCL_ASYNC_ERROR_HANDLING')},
                          ddp_history=all_history, ddp_parameter_checksum_difference=float(difference),
                          ddp_elapsed_seconds=time.time() - started)
            Path(args.sanity).write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
    except BaseException as exc:
        Path(args.sanity).with_name(f'ddp_failure_rank{rank}.json').write_text(json.dumps({'status': 'FAILED', 'error': repr(exc), 'history': history}, indent=2))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
