from __future__ import annotations

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


def read_ed_exact(path):
    # Per-file hint only: do not change OS/global disk settings. The source file
    # is C-contiguous float32 NPY; read only its first frame, bit-for-bit.
    with open(path, 'rb') as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
        version = np.lib.format.read_magic(handle)
        if version != (1, 0):
            raise ValueError(f'Unexpected derived NPY version: {version}')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        if shape != (50, 2, 6141, 3) or fortran or dtype != np.dtype('float32'):
            raise ValueError('Derived mesh layout does not match immutable cache contract')
        arr = np.fromfile(handle, dtype=dtype, count=2 * 6141 * 3)
        if arr.size != 2 * 6141 * 3:
            raise ValueError('Truncated ED frame')
        return arr.reshape(1, 2, 6141, 3)


class MeshDataset(Dataset):
    def __init__(self, manifest, split, ed_only=False):
        self.ed_only = ed_only
        self.table = pd.read_csv(manifest, dtype={'eid': str})
        self.table = self.table[self.table.split.eq(split)].reset_index(drop=True)
        if self.table.eid.duplicated().any():
            raise ValueError('Repeated subjects')

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        r = self.table.iloc[index]
        mesh = read_ed_exact(r.cache_path) if self.ed_only else np.load(r.cache_path, allow_pickle=False)
        if mesh.shape != ((1 if self.ed_only else 50), 2, 6141, 3):
            raise ValueError(f'Invalid derived cache {r.eid}')
        if not np.isfinite(mesh).all():
            raise ValueError(f'Nonfinite derived cache {r.eid}')
        return {'mesh': torch.from_numpy(mesh), 'eid': str(r.eid)}


class DistributedEvalSampler(Sampler):
    """No padding, so each subject is evaluated exactly once."""
    def __init__(self, dataset, rank=0, world_size=1):
        self.ids = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)
