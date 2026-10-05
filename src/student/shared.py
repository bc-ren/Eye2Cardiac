"""Standalone run configuration, immutable parent imports."""
import hashlib,json,os,random,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parent
S=json.loads((ROOT/'settings.json').read_text())
PREP=ROOT/S['prepared_name']
sys.path.append(S['parent'])
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n');q.replace(p)
def status(p):
    q=Path(p)/'STATUS.json';return json.loads(q.read_text())['status'] if q.exists() else None
def fingerprints():
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py')) if not any(s in p.parts for s in ['source_reference','__pycache__','code_snapshot'])}
    for name in ['settings.json','PROTOCOL.md','phenotypes.json']:
        files[name]=sha(ROOT/name)
    support=Path(S['parent'])
    for p in sorted(support.rglob('*.py')):
        files['student_base/'+str(p.relative_to(support))]=sha(p)
    files['student_base/config.json']=sha(support/'config.json')
    config=json.loads((support/'config.json').read_text())
    files['teacher/model.py']=sha(config['teacher_code'])
    retina=Path(S['retina_source'])
    for p in sorted(retina.rglob('*.py')):
        files['external_retizero/'+str(p.relative_to(retina))]=sha(p)
    return files
def paired_path():
    return PREP/'paired_SERVER_ONLY.parquet'
def seed_all(s):
    import numpy as np,torch
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def relocated_config(saved, runtime):
    """Keep saved numerical settings, but never restore obsolete private paths."""
    result=dict(saved)
    for key in ['root','prepared_directory','teacher_checkpoint','teacher_code','topology']:
        if key in runtime: result[key]=runtime[key]
    return result
@contextmanager
def stage(p,**kw):
    p=Path(p);p.mkdir(parents=True,exist_ok=False);save(p/'STATUS.json',dict(status='RUNNING',started=now(),**kw))
    try:
        yield p
        save(p/'STATUS.json',dict(status='COMPLETED',finished=now(),**kw))
    except BaseException as e:
        save(p/'STATUS.json',dict(status='INTERRUPTED' if isinstance(e,(KeyboardInterrupt,SystemExit)) else 'FAILED',error=repr(e),traceback=traceback.format_exc(),finished=now(),**kw));raise
