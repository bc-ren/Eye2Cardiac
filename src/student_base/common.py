"""Small, standalone utilities: no imports through deleted experiment directories."""
from pathlib import Path
import hashlib
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / 'config.json').read_text())
PREPARED = ROOT / C['prepared_directory']
SMOKE = ROOT / C['smoke_directory']
DEMO = ['age_cfp', 'sex_cfp', 'bsa_cfp']
LATENT = [f'latent_raw_{j:03d}' for j in range(96)]
VASC = ['temporal_cre_arteries', 'temporal_cre_veins', 'temporal_avr',
        'md_diam_arteries', 'md_diam_veins', 'vd_etdrs_full_arteries',
        'vd_etdrs_full_veins', 'mean_sparsity_etdrs_full',
        'md_tort_curv_arteries', 'md_tort_curv_veins']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + '\n')
    temp.replace(path)

def now():
    return datetime.now(timezone.utc).isoformat()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def fingerprints():
    files = list(ROOT.glob('*.py')) + list((ROOT / 'reference').glob('*.py')) + [ROOT / 'config.json', ROOT / 'PROTOCOL.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}

def design(x, median):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == 3 and np.isfinite(x[:, :2]).all()
    a, s = (x[:, 0] - 60) / 10, x[:, 1]
    missing = ~np.isfinite(x[:, 2])
    b = (np.where(missing, median, x[:, 2]) - 2) / .3
    return np.stack([a, s, b, a*a, a*s, b*s, a*b, missing.astype(float)], 1)

def fit_prior(x, y, alpha):
    median = np.nanmedian(x[:, 2])
    f = design(x, median)
    mean, std = f.mean(0), f.std(0)
    std = np.where(std > 1e-8, std, 1.)
    f = np.column_stack([np.ones(len(f)), (f - mean) / std])
    penalty = np.eye(f.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(f.T @ f + penalty, f.T @ y)
    assert np.isfinite(coef).all()
    return dict(median=median, mean=mean, std=std, coef=coef)
