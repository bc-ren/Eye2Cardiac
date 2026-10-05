from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib, json, traceback, importlib.util, sys
HERE = Path(__file__).resolve().parent
C = json.loads((HERE/'config.json').read_text())
ROOT = Path(C['root'])
STUDENT = Path(C['student'])
FIELDS11 = [f for f in C['fields13'] if f not in C['exclude']]
KEEP11 = [C['fields13'].index(f) for f in FIELDS11]
assert len(FIELDS11) == 11 and KEEP11 == [0,1,2,3,4,5,6,9,10,11,12]
ORDINARY = {
 'D0':['demo'], 'Student':['z0'], 'RetiZero':['retina'], 'Mesh11':['mesh11'],
 'D0+Student':['demo','z0'], 'D0+Student+RetiZero':['demo','z0','retina'],
 'D0+Student+Mesh11':['demo','z0','mesh11'],
 'D0+Student+RetiZero+Mesh11':['demo','z0','retina','mesh11'],
 'D0+ClinicalScore':['demo','clinical'], 'ClinicalScore+Student':['clinical','z0'],
 'ClinicalScore+RetiZero':['clinical','retina'],
 'ClinicalScore+Student+RetiZero':['clinical','z0','retina'],
 'D0+ClinicalScore+Student':['demo','clinical','z0'],
 'D0+ClinicalScore+RetiZero':['demo','clinical','retina'],
 'D0+ClinicalScore+Student+RetiZero':['demo','clinical','z0','retina'],
 'D0+ClinicalScore+Student+RetiZero+Mesh11':['demo','clinical','z0','retina','mesh11']}
RESIDUAL = {k.replace('D0+', 'D0-offset+', 1):v[1:] for k,v in ORDINARY.items()
            if k in ['D0+Student','D0+Student+RetiZero','D0+Student+Mesh11','D0+Student+RetiZero+Mesh11']}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 q.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n');q.replace(p)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def manifest(task):return ROOT/'prepared'/('recorded_timecovered_SERVER_ONLY.parquet' if task=='Diagnosis' else 'historical_country_SERVER_ONLY.parquet')
def run_name(seed):return f'cartesian_polar_lr1e-04_decay095_seed{seed}'
def state(p):
 f=Path(p)/'STATUS.json';return json.loads(f.read_text())['status'] if f.exists() else None
@contextmanager
def stage(p,**meta):
 p=Path(p);p.mkdir(parents=True,exist_ok=False);save(p/'STATUS.json',dict(status='RUNNING',started=now(),**meta))
 try:
  yield p
  save(p/'STATUS.json',dict(status='COMPLETED',finished=now(),**meta))
 except BaseException as e:
  save(p/'STATUS.json',dict(status='INTERRUPTED' if isinstance(e,(KeyboardInterrupt,SystemExit)) else 'FAILED',finished=now(),error=repr(e),traceback=traceback.format_exc(),**meta));raise
