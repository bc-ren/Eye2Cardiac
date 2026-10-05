"""Structured KD and physical constraints. No direct phenotype-output bypass."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from common import ROOT, load_module

reference = load_module('query_reference_objectives',ROOT/'reference/timebridge_objectives.py')
describe = reference.describe


class Geometry(nn.Module):
    def __init__(self, c):
        super().__init__()
        topo = np.load(c['topology'],allow_pickle=False)
        for side in ['LV','Myo']:
            faces = topo[side+'_faces']
            edges = topo[side+'_edges']
            # Uniform graph Laplacian; fixed topology, no patient fitted transform.
            src = np.concatenate([edges[:,0],edges[:,1]])
            dst = np.concatenate([edges[:,1],edges[:,0]])
            degree = np.bincount(src,minlength=6141).astype('float32')
            assert degree.min() > 0
            for key,val in [('faces',faces),('src',src),('dst',dst),('degree',degree)]:
                self.register_buffer(side+'_'+key,torch.from_numpy(val))

    def laplace(self,x,side):
        out = torch.zeros_like(x)
        src,dst = getattr(self,side+'_src'),getattr(self,side+'_dst')
        out.index_add_(2,src,x[:,:,dst])
        return x-out/getattr(self,side+'_degree')[None,None,:,None]

    def forward(self,p,t,ed,es):
        row = torch.arange(len(p),device=p.device)
        pp = torch.stack([p[row,ed],p[row,es]],1)
        tt = torch.stack([t[row,ed],t[row,es]],1)
        norms=[]; laps=[]
        for q,side in enumerate(['LV','Myo']):
            f = getattr(self,side+'_faces').long()
            a,b = pp[:,:,q,f],tt[:,:,q,f]
            na = F.normalize(torch.linalg.cross(a[...,1,:]-a[...,0,:],a[...,2,:]-a[...,0,:],dim=-1),dim=-1)
            nb = F.normalize(torch.linalg.cross(b[...,1,:]-b[...,0,:],b[...,2,:]-b[...,0,:],dim=-1),dim=-1)
            norms.append((1-(na*nb).sum(-1)).mean())
            laps.append(F.smooth_l1_loss(self.laplace(pp[:,:,q],side),self.laplace(tt[:,:,q],side)))
        return torch.stack(norms).mean(),torch.stack(laps).mean()


def physical(z,b,decoder,geometry,c):
    pmesh = checkpoint(decoder,z.float(),use_reentrant=False) if torch.is_grad_enabled() else decoder(z.float())
    tmesh = b['mesh_mm'].float()
    p = describe(pmesh,decoder)
    with torch.no_grad(): t = describe(tmesh,decoder)
    error = torch.linalg.vector_norm(pmesh-tmesh,dim=-1)
    geom = error.mean()/c['mesh_scale_mm']
    scale = z.new_tensor(c['phenotype_scales'])
    ph = F.smooth_l1_loss(p['ph']/scale,t['ph']/scale)
    curve = F.smooth_l1_loss(p['curves']/30,t['curves']/30)
    # beta=.05 means a 5 percentage-point normalized-volume deviation enters the linear region.
    pc = p['curves'][:,:,0]/p['ph'][:,:1].clamp_min(1)
    tc = t['curves'][:,:,0]/t['ph'][:,:1].clamp_min(1)
    rel = F.smooth_l1_loss(pc,tc,beta=.05)
    row = torch.arange(len(z),device=z.device)
    ix = (t['es'][:,None]+torch.arange(-2,3,device=z.device)[None]) % 50
    around_es = F.smooth_l1_loss(pc.gather(1,ix),tc.gather(1,ix),beta=.05)
    normal,lap = geometry(pmesh,tmesh,t['ed'],t['es'])
    # Same point correspondence; no per-patient registration/re-scaling.
    motion = F.smooth_l1_loss((pmesh-pmesh[row,t['ed']][:,None])/5,
                             (tmesh-tmesh[row,t['ed']][:,None])/5)
    invalid = (F.relu(1-p['curves'][:,:,0]).mean()+F.relu(p['curves'][:,:,0]-p['curves'][:,:,1]+1).mean())/100
    loss = (c['mesh_weight']*geom+c['phenotype_weight']*ph+c['curve_weight']*curve+
            c['relative_curve_weight']*rel+c['es_neighborhood_weight']*around_es+
            c['normal_weight']*normal+c['laplacian_weight']*lap+c['motion_weight']*motion+.1*invalid)
    stats = dict(mesh_mean_mm=error.mean(),mesh_mse_mm2=error.square().mean(),
        ED_mean_mm=error[row,t['ed']].mean(),ES_mean_mm=error[row,t['es']].mean(),
        LV_mean_mm=error[:,:,0].mean(),Myo_mean_mm=error[:,:,1].mean(),
        EF_MAE_pp=(p['ph'][:,3]-t['ph'][:,3]).abs().mean(),
        invalid_fraction=p['invalid'].float().mean(),
        relative_curve_mae=(pc-tc).abs().mean(),es_neighborhood_huber=around_es,
        selection_score=((p['ph'][:,[0,1,3,4]]-t['ph'][:,[0,1,3,4]]).abs()/scale[[0,1,3,4]]).mean()+.25*geom)
    return loss,{k:v.detach() for k,v in stats.items()},p,t


def relational(p,t):
    if len(p)<2: return p.sum()*0
    dp,dt = torch.pdist(p),torch.pdist(t.detach())
    valid = dt>1e-6
    if not valid.any(): return p.sum()*0
    return F.smooth_l1_loss(dp[valid]/dp[valid].mean().clamp_min(1e-6),dt[valid]/dt[valid].mean())


def conditional_contrast(p,t,demo,index,bank,c):
    bp,bd = bank['white'],bank['demo']
    p,t,bp = F.normalize(p,dim=1,eps=.1),F.normalize(t.detach(),dim=1,eps=.1),F.normalize(bp,dim=1,eps=.1)
    valid = ((demo[:,1,None]==bd[None,:,1]) & ((demo[:,0,None]-bd[None,:,0]).abs()<=c['negative_age_window_years']) &
             torch.isfinite(demo[:,2,None]) & torch.isfinite(bd[None,:,2]) &
             ((demo[:,2,None]-bd[None,:,2]).abs()<=c['negative_bsa_window_m2']))
    valid &= torch.arange(len(bp),device=p.device)[None] != index[:,None]
    valid &= (t@bp.T)<c['negative_max_cosine']
    ok = valid.any(1)
    if not ok.any(): return p.sum()*0,ok.float().mean()
    positive = (p*t).sum(1,keepdim=True)
    negative = (p@bp.T).masked_fill(~valid,-torch.inf)
    logits = torch.cat([positive,negative],1)[ok]/c['contrastive_temperature']
    return F.cross_entropy(logits,torch.zeros(int(ok.sum()),device=p.device,dtype=torch.long)),ok.float().mean()


def loss_for(model,out,b,decoder,geometry,c,bank=None,asynchronous=False,anchor=None):
    dt = b['delta_years']
    assert (dt.gt(0) if asynchronous else dt.eq(0)).all()
    if asynchronous:
        future = model.propagate(out,dt)
        z,res,prior = future['z1'],future['residual1'],future['prior1']
    else: z,res,prior = out['z0'],out['r0'],out['prior0']
    target = b['latent_std']-prior.detach()
    wp,wt = model.whiten(res),model.whiten(target)
    huber = F.smooth_l1_loss(wp,wt)
    cosine = sum((1-F.cosine_similarity(res[:,ids],target[:,ids],dim=1,eps=.1)).mean()
                 for ids in [model.shape_ids,model.motion_ids])/2
    kd = relational(wp,wt)
    phys,stats,p,t = physical(z,b,decoder,geometry,c)
    var = out['variance0']
    nll = .5*((wp-wt).square()/var+var.log()).mean()
    loss = phys+c['latent_weight']*huber+c['cosine_weight']*cosine+c['relational_weight']*kd+c['uncertainty_weight']*nll
    coverage = res.new_zeros(())
    if not asynchronous:
        assert bank is not None
        contrast,coverage = conditional_contrast(wp,wt,b['demo_cfp'],b['row_index'],bank,c)
        mask = b['vessel_mask']; count = mask.sum(1)
        mean = (b['vessels']*mask).sum(1)/count.clamp_min(1)
        vok = count>0
        vascular = F.smooth_l1_loss(out['vascular_prediction'][vok],mean[vok]) if vok.any() else res.sum()*0
        loss = loss+c['contrastive_weight']*contrast+c['vascular_aux_weight']*vascular
        if anchor is not None: loss = loss+c['anchor_weight']*F.mse_loss(out['z0'],anchor.detach())
    else:
        loss = loss+c['transport_regularization']*future['change'].square().mean()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite query/physical objective')
    stats.update(loss=loss.detach(),latent_huber=huber.detach(),cosine=cosine.detach(),
        relational=kd.detach(),gaussian_nll=nll.detach(),conditional_negative_coverage=coverage.detach())
    return loss,stats,p,t

from physical_chunks import scalar_chunks
_physical_full = physical
def physical(z,b,decoder,geometry,c):
    return scalar_chunks(_physical_full,z,b,decoder,geometry,c)
