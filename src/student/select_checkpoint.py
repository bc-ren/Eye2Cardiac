"""Record the completed validation-selected checkpoint; never inspect test."""
import argparse
import json
import torch
from shared import ROOT, S, save, sha, status


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    run=ROOT/'runs'/a.run
    assert status(run)=='COMPLETED'
    history=json.loads((run/'history.json').read_text())
    assert len(history)==S['max_epochs']==35
    assert all(r['train_seen']==1777 and r['async_seen']==0 for r in history)
    safe=[r for r in history if r['safety_pass']]
    chosen=min(safe,key=lambda r:r['validation']['selection_score'])
    checkpoint=run/'sync_only_best.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert ck['seed']==S['seed'], 'Checkpoint seed does not match the locked experiment'
    assert ck['epoch']==chosen['epoch'] and ck['phase']=='sync_only' and not ck['fallback']
    dest=ROOT/'selection';dest.mkdir(exist_ok=False)
    save(dest/'student_SELECTION.json',dict(run=a.run,selected_epoch=ck['epoch'],
         checkpoint_sha256=sha(checkpoint),validation=ck['validation'],
         test_used_for_selection=False,seed=S['seed'],trained_epochs=35,
         same_day_monitor_is_apparent=True))


if __name__=='__main__':main()
