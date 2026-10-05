"""Participant bootstrap of model-derived LV and geometric test metrics."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def correlation(a, b):
    aa, bb = a - a.mean(), b - b.mean()
    denominator = np.sqrt((aa * aa).sum() * (bb * bb).sum())
    return float((aa * bb).sum() / denominator) if denominator > 0 else np.nan


def metrics(y, p):
    error = p - y
    variance = ((y - y.mean()) ** 2).sum()
    return np.array([np.abs(error).mean(), np.sqrt((error ** 2).mean()), error.mean(),
                     correlation(y, p), correlation(rankdata(y), rankdata(p)),
                     1 - (error ** 2).sum() / variance if variance > 0 else np.nan])


def generate_report(source, output, repeats=2000, seed=2026):
    df = pd.read_csv(source, dtype={'eid': str})
    if len(df) != 4204 or df.eid.duplicated().any():
        raise ValueError('Expected 4,204 unique test subjects')
    out = Path(output)
    out.mkdir(exist_ok=False)
    names = ['MAE', 'RMSE', 'Bias', 'Pearson_r', 'Spearman_rho', 'R2']
    phenotype_data = {(v, p): (df['reference_' + p].to_numpy(), df[v + '_' + p].to_numpy())
                      for v in ('core', 'full') for p in ('LVEDV', 'LVESV', 'LVEF')}
    draws = {key: [] for key in phenotype_data}
    geom_cols = [c for c in df if c.endswith(('_mean_mm', '_rmse_mm', '_p95_mm'))]
    geom = df[geom_cols].to_numpy()
    if not np.isfinite(geom).all() or not all(np.isfinite(y).all() and np.isfinite(p).all() for y, p in phenotype_data.values()):
        raise ValueError('Nonfinite results; no silent participant exclusions permitted')
    rng = np.random.default_rng(seed)
    geom_draws = []
    for iteration in range(repeats):
        ids = rng.integers(0, len(df), size=len(df))
        geom_draws.append(geom[ids].mean(0))
        for key, (y, p) in phenotype_data.items():
            draws[key].append(metrics(y[ids], p[ids]))
    rows = []
    for (variant, phenotype), (y, p) in phenotype_data.items():
        estimate = metrics(y, p)
        ci = np.nanquantile(draws[(variant, phenotype)], [.025, .975], axis=0)
        for j, name in enumerate(names):
            rows.append({'variant': variant, 'phenotype': phenotype, 'metric': name,
                         'value': estimate[j], 'ci_low': ci[0, j], 'ci_high': ci[1, j],
                         'unit': ('pp' if phenotype == 'LVEF' else 'mL') if name in ('MAE', 'RMSE', 'Bias') else 'dimensionless',
                         'n': len(df), 'bootstrap_valid_repeats': int(np.isfinite(np.asarray(draws[(variant, phenotype)])[:, j]).sum())})
    pd.DataFrame(rows).to_csv(out / 'LV_phenotype_metrics_with_95CI.csv', index=False)
    gci = np.quantile(geom_draws, [.025, .975], axis=0)
    pd.DataFrame({'metric': geom_cols, 'value': geom.mean(0), 'ci_low': gci[0], 'ci_high': gci[1],
                  'n': len(df), 'unit': 'mm'}).to_csv(out / 'geometry_metrics_with_95CI.csv', index=False)
    (out / 'protocol.json').write_text(json.dumps({'status': 'COMPLETED', 'seed': seed,
        'bootstrap_unit': 'participant, all frames and both surfaces together', 'repeats': repeats,
        'interval': 'percentile 95%', 'core_full_use_identical_resamples': True,
        'n': len(df), 'source': str(source), 'invalid_or_undefined_correlations': 'reported through valid bootstrap count',
        'clinical_wall_thickness': 'not reported', 'other_CMR12_chambers': 'not reconstructed'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    generate_report(a.source, a.output)
