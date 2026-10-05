"""Immutable derived LV/Myo caches and training-only assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

G = {}


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def init_worker(topology, cache):
    z = np.load(topology, allow_pickle=False)
    G['ids'] = np.stack([np.flatnonzero(z['full_labels'] == q) for q in (1, 2)])
    maps = []
    for q, ids in zip((1, 2), G['ids']):
        remap = np.full(len(z['full_labels']), -1)
        remap[ids] = np.arange(len(ids))
        maps.append(remap[z['full_faces'][z['cell_labels'] == q]])
    assert np.array_equal(*maps)
    G['faces'] = maps[0]
    G['cache'] = Path(cache)


def volume(x, faces):
    # Float64 accumulation; avoid creating all 50x12k triangles in float64.
    out = []
    for frame in x:
        tri = frame[faces].astype(np.float64)
        out.append(np.sum(tri[:, 0] * np.cross(tri[:, 1], tri[:, 2])) / 6000.)
    return np.asarray(out)


def prepare_one(item):
    ix, eid, split, source = item
    path = G['cache'] / f'{ix // 512:04d}' / f'{eid}.npy'
    if path.exists():
        raise FileExistsError(f'Refusing existing derived cache: {path}')
    with np.load(source, allow_pickle=False) as z:
        raw = z['mesh_mm']
    if raw.shape != (50, 27034, 3) or not np.isfinite(raw).all():
        raise ValueError(f'Invalid source mesh for EID {eid}')
    mesh = np.asarray(raw[:, G['ids'], :], dtype=np.float32)
    lv_signed = volume(mesh[:, 0], G['faces'])
    if (lv_signed <= 1e-6).any():
        raise ValueError(f'Inconsistent LV signed orientation/volume for EID {eid}')
    ed = int(lv_signed.argmax())
    es = int(lv_signed.argmin())
    mesh = np.ascontiguousarray(np.roll(mesh, -ed, axis=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('xb') as f:
        np.save(f, mesh, allow_pickle=False)
    path.with_suffix('.tmp').replace(path)
    record = {'eid': str(eid), 'split': split, 'cache_path': str(path),
              'source_mesh_path': source, 'original_ed_index': ed,
              'es_index': (es - ed) % 50, 'reference_edv_ml': float(lv_signed.max()),
              'reference_esv_ml': float(lv_signed.min()),
              'reference_ef_pp': float(100 * (1 - lv_signed.min() / lv_signed.max())),
              'cache_sha256': hashlib.sha256(mesh.tobytes()).hexdigest()}
    return ix, record, mesh[0].astype(np.float64) if split == 'train' else None


def fps(points, count):
    selected = np.empty(count, np.int64)
    selected[0] = int(np.argmin(np.linalg.norm(points - points.mean(0), axis=1)))
    distances = np.full(len(points), np.inf)
    for i in range(1, count):
        distances = np.minimum(distances, np.sum((points - points[selected[i - 1]]) ** 2, 1))
        distances[selected[:i]] = -1
        selected[i] = int(distances.argmax())
    return selected


def assets(mean, topology, target):
    z = np.load(topology, allow_pickle=False)
    arrays = {'template_mm': mean.astype(np.float32)}
    for q, name in zip((1, 2), ('LV', 'Myo')):
        ids = np.flatnonzero(z['full_labels'] == q)
        remap = np.full(len(z['full_labels']), -1)
        remap[ids] = np.arange(len(ids))
        f = remap[z['full_faces'][z['cell_labels'] == q]]
        arrays[name + '_faces'] = f
        arrays[name + '_global_ids'] = ids
        e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
        arrays[name + '_edges'] = e
        neighbors = [set() for _ in ids]
        for a, b in e:
            neighbors[a].add(b)
            neighbors[b].add(a)
        # Full-resolution graph Laplacian uses all exact one-ring neighbors.
        max_deg = max(map(len, neighbors))
        adj = np.repeat(np.arange(len(ids))[:, None], max_deg, axis=1)
        weights = np.zeros_like(adj, dtype=np.float32)
        for j, ns in enumerate(neighbors):
            ns = sorted(ns)
            adj[j, :len(ns)] = ns
            weights[j, :len(ns)] = 1 / len(ns)
        arrays[name + '_lap_indices'] = adj
        arrays[name + '_lap_weights'] = weights
        p = mean[q - 1].copy()
        for level, n in enumerate((6141, 1536, 384, 96, 24)):
            if level:
                down = fps(p, n)
                new = p[down]
                dist, idx = cKDTree(new).query(p, k=4)
                w = 1 / np.maximum(dist, 1e-8)
                w /= w.sum(1, keepdims=True)
                arrays[f'{name}_down{level}'] = down
                arrays[f'{name}_up_idx{level}'] = idx.astype(np.int64)
                arrays[f'{name}_up_w{level}'] = w.astype(np.float32)
                p = new
            # Template-fixed kNN neighborhoods: explicit, deterministic ordering.
            arrays[f'{name}_spiral{level}'] = cKDTree(p).query(p, k=9)[1].astype(np.int64)
    np.savez_compressed(target, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-manifest', type=Path, required=True,
                        help='Authorized CSV with eid, split and mesh_path columns')
    parser.add_argument('--source-topology', type=Path, required=True,
                        help='Authorized original topology NPZ; not a generated LV/Myo topology')
    parser.add_argument('--expected-manifest-sha256', required=True,
                        help='Expected SHA-256 of the exact local source manifest')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    prep = root / 'preparation'
    prep.mkdir(exist_ok=False)
    cache = root / 'cache'
    cache.mkdir(exist_ok=False)
    manifest = args.source_manifest
    topology = args.source_topology
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if sha != args.expected_manifest_sha256.lower():
        raise ValueError('Source manifest hash mismatch')
    table = pd.read_csv(manifest, dtype={'eid': str})
    if table.eid.duplicated().any():
        raise ValueError('Repeated EID in source manifest')
    if table.split.value_counts().to_dict() != {'train': 46969, 'test': 4204, 'val': 2112}:
        raise ValueError('Source counts differ')
    for p in table.mesh_path:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    items = [(i, r.eid, r.split, r.mesh_path) for i, r in enumerate(table.itertuples())]
    summary = {'status': 'PREPARING', 'source_sha256': sha, 'total': len(items),
               'workers': args.workers, 'done': 0, 'start_time': time.time(),
               'mean_accumulation': 'float64_in_manifest_order',
               'changed_preprocessing': 'explicit_shared_ED_circular_shift_and_LV_Myo_extraction'}
    save_json(prep / 'status.json', summary)
    mean_sum = np.zeros((2, 6141, 3), dtype=np.float64)
    records = []
    train_count = 0
    try:
        with mp.get_context('spawn').Pool(args.workers, initializer=init_worker,
                                         initargs=(str(topology), str(cache))) as pool:
            for ix, record, ed in pool.imap(prepare_one, items, chunksize=1):
                records.append(record)
                if ed is not None:
                    mean_sum += ed
                    train_count += 1
                if (ix + 1) % 100 == 0:
                    summary.update(done=ix + 1, elapsed_seconds=time.time() - summary['start_time'])
                    save_json(prep / 'status.json', summary)
                    print(json.dumps({k: summary[k] for k in ('status', 'done', 'total', 'elapsed_seconds')}), flush=True)
        assert train_count == 46969
        mean = mean_sum / train_count
        frame = pd.DataFrame(records)
        frame.to_csv(prep / 'manifest.csv', index=False)
        (root / 'assets').mkdir(exist_ok=False)
        np.save(root / 'assets/train_mean_ed_mm.npy', mean)
        assets(mean, topology, root / 'assets/lv_myo_topology.npz')
        summary.update(status='COMPLETED', done=len(items), train_count=train_count,
                       elapsed_seconds=time.time() - summary['start_time'],
                       ed_shift_counts=frame.original_ed_index.value_counts().sort_index().to_dict(),
                       topology_sha256=hashlib.sha256((root / 'assets/lv_myo_topology.npz').read_bytes()).hexdigest(),
                       derived_manifest_sha256=hashlib.sha256((prep / 'manifest.csv').read_bytes()).hexdigest())
        save_json(prep / 'status.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        summary.update(status='FAILED', error=repr(exc), done=len(records))
        save_json(prep / 'status.json', summary)
        raise


if __name__ == '__main__':
    main()
