"""Preserve observed pre-CFP clinical values; impute within each head fold later."""
import numpy as np,pandas as pd
from runlib import *

def main():
 assert state(ROOT/'preflight')=='COMPLETED'
 with stage(ROOT/'clinical_raw') as out:
  reader=module('clinical_reader',HERE/'clinical_reader.py')
  reader.FIELD_IDS=('31','53','93','94','1239','2443','4079','4080','6153','6177','20002','20116','21000','30690','30760')
  d=pd.read_parquet(manifest('Diagnosis'));r=reader.read_target_rows(Path(C['raw_csv']),d.eid.astype(str).to_numpy())
  visits=reader.assessment_dates(r);retina=pd.to_datetime(d.retina_date).to_numpy(dtype='datetime64[D]')
  def numeric(fields):
   v,m=reader.numeric_by_visit(r,fields);return reader.latest_prior(v,m,visits,retina)
  columns=[]
  for key in ['age_cfp','sex_cfp']:
   columns.append(np.where(d[key+'_observed'].to_numpy(bool),d[key].to_numpy(float),np.nan))
  for fields in [('30690',),('30760',),('4080','93'),('4079','94')]:
   v,m=numeric(fields);columns.append(np.where(m,v,np.nan))
  v,m=reader.categorical_presence_by_visit(r,('2443',),{1});dq,mq=reader.latest_prior(v,m,visits,retina)
  ds,ms=reader.any_prior_condition(r,visits,retina,reader.SELF_REPORTED['diabetes'])
  columns.append(np.where(mq|ms,(dq==1)|ds,np.nan))
  sm,km=numeric(('20116',));tb,kt=numeric(('1239',))
  columns.append(np.where(km|kt,np.where(km,sm==2,tb>0),np.nan))
  x=np.column_stack(columns);np.save(out/'observed_inputs_SERVER_ONLY.npy',x)
  save(out/'PROVENANCE.json',dict(order=['age_cfp','sex_cfp','total_chol_mmol_l','hdl_mmol_l','sbp_mmhg','dbp_mmhg','diabetes','current_smoker'],n=len(d),missing_per_column=np.isnan(x).sum(0).tolist(),imputation='none; fit in each head-training fold',latest_visit_at_or_before_CFP=True,source_manifest_sha256=sha(manifest('Diagnosis')),source_formula_sha256=sha(HERE/'clinical_formula.py'),raw_csv_bytes=Path(C['raw_csv']).stat().st_size))
if __name__=='__main__':main()
