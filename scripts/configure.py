"""Stage portable source into a new private workspace; never launch a job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from string import Template

REPO=Path(__file__).resolve().parents[1]
KEYS={'STUDENT_PREPARED','ANATOMY_CACHE','PRIOR_SOURCE','PRIOR_PREPARED',
      'RETIZERO_SOURCE','RETIZERO_CHECKPOINT','TEACHER_CHECKPOINT','TEACHER_TOPOLOGY',
      'FULL_TOPOLOGY','BASAL_RING_IDS','TEACHER_DATA','DOWNSTREAM_PREPARED','CLINICAL_CSV'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def resolve_template(value,context):
    """Substitute parsed values, never interpolate unescaped text into JSON."""
    if isinstance(value,str):return Template(value).substitute(context)
    if isinstance(value,list):return [resolve_template(x,context) for x in value]
    if isinstance(value,dict):return {k:resolve_template(v,context) for k,v in value.items()}
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();raw=json.loads(a.paths.read_text())
    if set(raw)!=KEYS:raise ValueError(f'Path keys differ: missing={KEYS-set(raw)}, extra={set(raw)-KEYS}')
    paths={k:Path(v).expanduser().resolve() for k,v in raw.items()}
    missing=[k for k,v in paths.items() if not v.exists()]
    if missing:raise FileNotFoundError('Supply real, authorized local inputs for: '+', '.join(missing))
    out=a.output.expanduser().resolve()
    if out.exists():raise FileExistsError('Refusing to overwrite an existing workspace: '+str(out))
    if out==REPO or REPO in out.parents:raise ValueError('Use a private workspace outside the public repository')
    for v in paths.values():
        if out==v or out in v.parents or v in out.parents:
            raise ValueError('Workspace must not contain or be nested inside an input path')
    manifest_names=[x+'_SERVER_ONLY.parquet' for x in
        ['same_day_train','same_day_validation','same_day_test','asynchronous_train',
         'asynchronous_validation','asynchronous_test','joint_train','image_manifest']]
    files=[paths['STUDENT_PREPARED']/n for n in manifest_names]
    files += [paths['STUDENT_PREPARED']/'fold_0'/n for n in ['prior_raw.npz','scalers.npz','residual_whitening.npz','AUDIT.json']]
    files += [paths['DOWNSTREAM_PREPARED']/n for n in ['recorded_timecovered_SERVER_ONLY.parquet','historical_country_SERVER_ONLY.parquet']]
    files += [paths[k] for k in ['RETIZERO_CHECKPOINT','TEACHER_CHECKPOINT','TEACHER_TOPOLOGY','FULL_TOPOLOGY','BASAL_RING_IDS']]
    for file in files:
        if not file.is_file():raise FileNotFoundError(file)
    if a.check_only:
        print('Required input paths exist; no source staged and no scientific validity claimed.')
        return
    context={k:str(v) for k,v in paths.items()}|{'RUN':str(out)}
    mapping=[('student.json','student/settings.json'),('student_model.json','student_base/config.json'),
             ('phenotypes.json','student/phenotypes.json'),('downstream.json','student/downstream/config.json')]
    resolved={dest:resolve_template(json.loads((REPO/'configs'/source).read_text()),context) for source,dest in mapping}
    # YAML path values use JSON quoting, valid YAML even when paths contain ':' or spaces.
    yaml_context={k:json.dumps(str(v)) for k,v in paths.items()}
    yaml_context['TEACHER_MANIFEST']=json.dumps(str(paths['TEACHER_DATA']/'preparation/manifest.csv'))
    teacher_yaml=Template((REPO/'configs/teacher.yaml').read_text()).substitute(yaml_context)
    # Validate all templates before creating anything. No late JSON escaping failure.
    out.mkdir(parents=True,exist_ok=False)
    write(out/'STATUS.json',{'status':'CONFIGURING','training_started':False})
    for source,destination in [('teacher','teacher'),('student_base','student_base'),('student','student'),('downstream','student/downstream')]:
        shutil.copytree(REPO/'src'/source,out/destination)
    for dest,value in resolved.items():write(out/dest,value)
    (out/'teacher/config.yaml').write_text(teacher_yaml)
    (out/'student/prepared').symlink_to(paths['STUDENT_PREPARED'],target_is_directory=True)
    (out/'student/downstream/prepared').symlink_to(paths['DOWNSTREAM_PREPARED'],target_is_directory=True)
    shutil.copy2(REPO/'docs/protocol.md',out/'student/PROTOCOL.md')
    hashes={str(file):sha(file) for file in files}
    write(out/'student/INPUT_HASHES.json',hashes)
    write(out/'PATHS.private.json',context)
    write(out/'STATUS.json',{'status':'CONFIGURED_NOT_TRAINED','training_started':False,
          'source_hashes':{str(f.relative_to(REPO)):sha(f) for f in sorted(REPO.rglob('*')) if f.is_file() and '__pycache__' not in f.parts},
          'input_hashes':hashes,'private_data_are_external':True})
    print('Configured source:',out)
    print('No training or evaluation was started. Run the documented preflight next.')


if __name__=='__main__':main()
