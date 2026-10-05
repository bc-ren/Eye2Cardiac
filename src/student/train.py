"""Single-stage whole-pool random DDP training; no source-ratio sampler."""
import argparse
import shutil
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from shared import *
from dataset import Cache, CachedPaired, global_batches, ShardBatches
from student import make_models, args_of
from data import worker_init, device_batch
from engine import bank_for
from staged_reference_train import cfg, loader, combine_validation, evaluate, optimizer, require_finite_grads
from joint_system import JointSystem
from retizero_backbone import frozen_hash, assert_scope


def probe(batch, name):
    torch.cuda.set_device(0)
    torch.set_num_threads(S['cpu_threads'])
    seed_all(S['seed'])
    c = cfg()
    with stage(ROOT/'capacity'/name, batch=batch, visible=os.environ.get('CUDA_VISIBLE_DEVICES')) as out:
        cache = Cache()
        ds = CachedPaired('train', 'same_day', cache)
        st = CachedPaired('train', 'same_day', cache)
        m, dec, geo = make_models(c)
        system = JointSystem(m, dec, geo, c, bank_for(m, st)).train()
        opt = optimizer(m, c)
        free, total = torch.cuda.mem_get_info()
        other = max(0, total-free-torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        stress = []
        frozen_before = frozen_hash(m.backbone.vit)
        assert_scope(m.backbone)
        for view in ['same_day']:
            if view == 'random_mixture':
                ix = np.random.default_rng(S['seed']).permutation(len(ds))[:batch]
            else:
                ix = ds.table.index[ds.table['view'].eq(view) & ds.table.L_image_path.notna() & ds.table.R_image_path.notna()].to_numpy()[:batch]
            b = device_batch(next(iter(loader(ds, dict(c,batch_size=batch,workers=0),ix))), 'cuda')
            for _ in range(2):
                before = {i:m.backbone.vit.blocks[i].norm1.weight.detach().clone() for i in [20,21,22,23]}
                opt.zero_grad(set_to_none=True)
                loss = system(b)
                loss.backward()
                require_finite_grads(m, True)
                torch.nn.utils.clip_grad_norm_(m.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                opt.step()
                assert all(not torch.equal(before[i],m.backbone.vit.blocks[i].norm1.weight) for i in before)
                assert frozen_hash(m.backbone.vit)==frozen_before
            stress.append(dict(scenario=view,n=len(ix),same_day=int((~b['is_async']).sum()),asynchronous=int(b['is_async'].sum()),loss=float(loss.detach())))
            del b
        system.eval()
        torch.save(m.export(),out/'roundtrip.pt')
        state={k:v.clone() for k,v in m.export().items()}
        m.restore(torch.load(out/'roundtrip.pt',weights_only=True))
        for key,value in state.items():torch.testing.assert_close(value,m.export()[key],atol=0,rtol=0)
        sv=CachedPaired('validation','same_day',cache);av=CachedPaired('validation','asynchronous',cache)
        val=combine_validation(evaluate(m,dec,geo,sv,c,2,max_batches=1),evaluate(m,dec,geo,av,c,2,max_batches=1))
        save(out/'RESULT.json',dict(batch=batch,seconds=time.time()-start,within_budget=torch.cuda.max_memory_reserved()+other+512*1024**2<total*S['batch_memory_fraction'],
             peak_reserved=torch.cuda.max_memory_reserved(),total=total,stress_scenarios=stress,
             last4_visual_blocks_have_gradients=True,last4_blocks_updated=True,frozen_backbone_unchanged=True,strict_roundtrip=True,
             tiny_evaluation=val,code=fingerprints()))


def run(seed,batch,name,smoke=False):
    assert seed == S['seed'], 'This release locks the original seed; use a separately documented protocol for new seeds'
    if not smoke:
        audit=json.loads((ROOT/'PREFLIGHT_COMPLETED.json').read_text())
        assert audit['status']=='COMPLETED' and audit['code']==fingerprints()
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
    torch.cuda.set_device(rank);dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    torch.set_num_threads(S['cpu_threads']);seed_all(seed);c=cfg()
    out=ROOT/('ddp_smoke' if smoke else 'runs')/name
    if rank==0:
        out.mkdir(parents=True,exist_ok=False);save(out/'STATUS.json',dict(status='RUNNING',started=now()))
        snap=out/'code_snapshot';snap.mkdir()
        for p in [*ROOT.glob('*.py'),ROOT/'settings.json',ROOT/'PROTOCOL.md']:shutil.copy2(p,snap/p.name)
        save(out/'CONFIG.json',dict(settings=S,parent=c,seed=seed,per_gpu_batch=batch,global_group_batch=2*batch))
        save(out/'PROVENANCE.json',dict(command=sys.argv,code=fingerprints(),teacher_hash=sha(c['teacher_checkpoint']),retizero_initial_hash=sha(S['retina_checkpoint']),
             manifest_hash=sha(PREP/'same_day_train_SERVER_ONLY.parquet'),torch=torch.__version__,cuda=torch.version.cuda,
             DDP=True,last4_visual_finetuning=True,training_mode='same_day_only_random_without_replacement',no_fixed_source_ratio=True,
             no_sampler_padding=True,no_checkpoint_resume=True,git_commit=None,git_reason='standalone source snapshot',
             nondeterminism='CUDA reductions; per-rank within-domain relational KD',test_used_for_selection=False,same_day_monitor_is_apparent=True))
    dist.barrier();start=time.time()
    try:
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
        cache=Cache();ds=CachedPaired('train','same_day',cache);st=CachedPaired('train','same_day',cache)
        sv=CachedPaired('validation','same_day',cache);av=CachedPaired('validation','asynchronous',cache)
        m,dec,geo=make_models(c);system=JointSystem(m,dec,geo,c,bank_for(m,st))
        model=DDP(system,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
        opt=optimizer(m,c);schedule=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=S['lr_decay_per_epoch'])
        history=[];best=float('inf');epochs=2 if smoke else S['max_epochs']
        for epoch in range(1,epochs+1):
            batches=global_batches(len(ds),batch,seed,1000*epoch)
            if smoke:batches=batches[:2]
            if rank==0:np.savez_compressed(out/f'sync_only_{epoch:03d}_draws_SERVER_ONLY.npz',indices=np.concatenate(batches),sizes=np.array(list(map(len,batches))))
            dl=DataLoader(ds,batch_sampler=ShardBatches(batches,rank),num_workers=S['workers'],pin_memory=True,
                          worker_init_fn=worker_init,persistent_workers=S['workers']>0,generator=torch.Generator().manual_seed(seed+epoch+rank))
            model.train();sums=torch.zeros(3,device='cuda',dtype=torch.float64);lr=opt.param_groups[0]['lr']
            for step,b in enumerate(dl):
                b=device_batch(b,'cuda');n=len(b['images']);opt.zero_grad(set_to_none=True)
                loss=model(b);(loss*(world*n/len(batches[step]))).backward()
                if step==0:require_finite_grads(m,True)
                grad=torch.nn.utils.clip_grad_norm_(m.parameters(),c['gradient_clip'],error_if_nonfinite=True);opt.step()
                sums+=torch.stack([loss.detach().double()*n,torch.tensor(float(n),device='cuda'),b['is_async'].double().sum()])
                if rank==0:
                    progress=dict(phase='sync_only',epoch=epoch,epochs=epochs,global_epoch=epoch,lr=lr,retizero_lr=lr,
                         step=step+1,steps=len(batches),loss=float(loss.detach()),gradient_norm=float(grad),
                         local_same_day=int((~b['is_async']).sum()),local_asynchronous=int(b['is_async'].sum()),
                         no_fixed_source_ratio=True,per_gpu_batch=batch,seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
                    save(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
            dist.all_reduce(sums)
            if not smoke:
                assert int(sums[1])==len(ds)==1777 and int(sums[2])==0
            vs=evaluate(m,dec,geo,sv,c,batch,rank,world,max_batches=1 if smoke else None)
            va=evaluate(m,dec,geo,av,c,batch,rank,world,max_batches=1 if smoke else None)
            val=combine_validation(vs,va);safe=val['invalid_fraction']<=c['max_invalid_fraction']
            improved=safe and val['selection_score']<best
            if improved:best=val['selection_score']
            schedule.step()
            if rank==0:
                state=dict(model=m.export(),phase='sync_only',arm='cartesian_polar',seed=seed,lr=lr,epoch=epoch,global_epoch=epoch,
                           validation=val,config=c,fallback=False,same_day_monitor_is_apparent=True)
                torch.save(dict(state,optimizer=opt.state_dict(),scheduler=schedule.state_dict()),out/'latest.pt')
                if improved:torch.save(state,out/'sync_only_best.pt')
                row=dict(phase='sync_only',epoch=epoch,global_epoch=epoch,lr=lr,next_lr=opt.param_groups[0]['lr'],validation=val,
                         train_loss=float(sums[0]/sums[1]),train_seen=int(sums[1]),sync_seen=int(sums[1]-sums[2]),async_seen=int(sums[2]),
                         safety_pass=bool(safe),seconds=time.time()-start)
                history.append(row);save(out/'history.json',history);print(json.dumps(row),flush=True)
            dist.barrier()
        if rank==0:
            assert (out/'sync_only_best.pt').is_file(),'No safe positive-epoch checkpoint; do not fall back to another run'
            ck=torch.load(out/'sync_only_best.pt',map_location='cpu',weights_only=False)
            save(out/'sync_only_RESULT.json',dict(validation=ck['validation'],selected_epoch=ck['epoch'],selected_global_epoch=ck['global_epoch'],
                 fallback=False,checkpoint_sha256=sha(out/'sync_only_best.pt'),seconds=time.time()-start))
            save(out/'STATUS.json',dict(status='COMPLETED',finished=now(),smoke=smoke))
        dist.barrier()
    except BaseException as e:
        if rank==0:save(out/'STATUS.json',dict(status='INTERRUPTED' if isinstance(e,(KeyboardInterrupt,SystemExit)) else 'FAILED',error=repr(e),traceback=traceback.format_exc(),finished=now()))
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--batch',type=int,required=True)
    p.add_argument('--seed',type=int,default=S['seed']);p.add_argument('--name',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.probe:probe(a.batch,a.name)
    else:run(a.seed,a.batch,a.name,a.smoke)
