"""Validate privately staged disease manifests without constructing new splits.

Checks are retained from the original prepare.py. The historical manifest-copy,
server inspection and three-seed selection assertion are intentionally absent.
"""
import json
import numpy as np
import pandas as pd
from runlib import *


def main():
 with stage(ROOT/'preflight') as out:
  assert sha(HERE/'metrics.py') == sha(C['metric_source'])
  st = pd.read_parquet(STUDENT/'prepared/joint_train_SERVER_ONLY.parquet')
  assert st.eid.is_unique
  train_ids = set(st.eid.astype(str))
  test_ids = set()
  for view in ('same_day', 'asynchronous'):
   test_table = pd.read_parquet(STUDENT/'prepared'/(view+'_test_SERVER_ONLY.parquet'))
   assert test_table.eid.is_unique
   ids = set(test_table.eid.astype(str))
   assert not ids & test_ids, 'Student test views overlap'
   test_ids |= ids
  assert not train_ids & test_ids
  reference = None
  info = {}
  for task in C['tasks']:
   d = pd.read_parquet(manifest(task))
   assert d.eid.is_unique
   if reference is not None:
    assert reference.eid.astype(str).tolist() == d.eid.astype(str).tolist()
    assert pd.to_datetime(reference.retina_date).equals(pd.to_datetime(d.retina_date))
    np.testing.assert_allclose(reference.age_cfp, d.age_cfp)
   reference = d
   eligible = d.pre_eligible.to_numpy(bool) if task == 'Diagnosis' else d.incident_eligible.to_numpy(bool)
   y = d.pre_y.to_numpy(bool) if task == 'Diagnosis' else d.event.to_numpy(bool)
   dev = d.original_split.isin(['train', 'validation']).to_numpy() & eligible
   test = d.original_split.eq('test').to_numpy() & d.paired_view.ne('unpaired').to_numpy() & eligible
   external = (d.original_split.eq('evaluation_only').to_numpy() &
               d.paired_view.eq('unpaired').to_numpy() & eligible &
               d.matched_augmentation_role.eq('none').to_numpy())
   if task == 'Incident':
    external &= d.in_incident_extension.to_numpy(bool)
   aug = d.matched_augmentation_role.ne('none').to_numpy()
   for mask in (test, external):
    ids = set(d.loc[mask, 'eid'].astype(str))
    assert not ids & train_ids
    assert not ids & set(d.loc[dev | aug, 'eid'].astype(str))
   assert not set(d.loc[dev | aug, 'eid'].astype(str)) & test_ids
   fold = d['pre_fold' if task == 'Diagnosis' else 'surv_fold'].to_numpy(int)
   assert set(fold[dev]) == set(range(5)), 'Development records require five fixed folds'
   for j in range(5):
    assert y[dev & (fold == j)].any() and (~y[dev & (fold == j)]).any()
   info[task] = {key: dict(n=int(mask.sum()), positive_or_event=int(y[mask].sum()))
                 for key, mask in [('development', dev), ('internal_test', test),
                                   ('external_holdout', external), ('augmentation', aug)]}
   info[task]['manifest_sha256'] = sha(manifest(task))
  joined = st.assign(eid=st.eid.astype(str)).merge(
   reference.assign(eid=reference.eid.astype(str)), on='eid', suffixes=('_student', '_head'))
  same = pd.to_datetime(joined.retina_date_student) == pd.to_datetime(joined.retina_date_head)
  np.testing.assert_allclose(joined.loc[same, 'age_model'], joined.loc[same, 'age_cfp_head'])
  images = pd.read_parquet(STUDENT/'prepared/image_manifest_SERVER_ONLY.parquet')
  paths = set(reference.L_image_path.dropna()) | set(reference.R_image_path.dropna())
  assert not paths - set(images.image_path)
  save(out/'AUDIT.json', dict(tasks=info, student_train_n=len(train_ids),
       all_test_student_train_overlap=0, age_anchor='exact head CFP visit',
       different_student_head_visits=int((~same).sum()), same_visit_age_equal=True,
       fields11=FIELDS11, excluded=C['exclude'], formal_survival_coverage_verified=False,
       code_sha256=sha(__file__), config_sha256=sha(HERE/'config.json'),
       cohort_construction='User-supplied original locked manifests; not regenerated'))
  print(json.dumps(info), flush=True)


if __name__ == '__main__':
 main()
