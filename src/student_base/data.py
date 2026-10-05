"""Original Linux loader utilities; no inherited augmentation or sampler."""
import os, random
import numpy as np
import torch
PARENT_CPU_AFFINITY = set(os.sched_getaffinity(0))

def device_batch(b, device): return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def worker_init(_):
    # The host OpenMP initialization otherwise pins every worker to CPU 0/64.
    os.sched_setaffinity(0, PARENT_CPU_AFFINITY)
    s = torch.initial_seed() % 2**32; random.seed(s); np.random.seed(s)

