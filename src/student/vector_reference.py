"""Patient-reduced baseline objectives, explicit temporal attenuation."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from objectives import describe,physical
from student import args_of
from shared import S

def average(x):return x.flatten(1).mean(1)
def huber(x,y,beta=1.):return average(F.smooth_l1_loss(x,y,reduction='none',beta=beta))
def weights(dt):
    assert dt.ndim==1 and torch.isfinite(dt).all() and (dt>=0).all()
    return torch.where(dt>0,S['async_coefficient']*torch.exp2(-dt/S['time_half_life_years']),torch.ones_like(dt))

def _physical_vector_full(z,b,decoder,geometry,c):
    pred=checkpoint(decoder,z.float(),use_reentrant=False) if torch.is_grad_enabled() else decoder(z.float())
    truth=b['mesh_mm'].float();p=describe(pred,decoder)
    with torch.no_grad():t=describe(truth,decoder)
    error=torch.linalg.vector_norm(pred-truth,dim=-1)
    geom=average(error)/c['mesh_scale_mm'];scale=z.new_tensor(c['phenotype_scales'])
    pc=p['curves'][:,:,0]/p['ph'][:,:1].clamp_min(1);tc=t['curves'][:,:,0]/t['ph'][:,:1].clamp_min(1)
    row=torch.arange(len(z),device=z.device);ix=(t['es'][:,None]+torch.arange(-2,3,device=z.device)[None])%50
    es=huber(pc.gather(1,ix),tc.gather(1,ix),.05)
    pp=torch.stack([pred[row,t['ed']],pred[row,t['es']]],1);tt=torch.stack([truth[row,t['ed']],truth[row,t['es']]],1)
    normals=[];laps=[]
    for q,side in enumerate(['LV','Myo']):
        f=getattr(geometry,side+'_faces').long();a,bm=pp[:,:,q,f],tt[:,:,q,f]
        na=F.normalize(torch.linalg.cross(a[...,1,:]-a[...,0,:],a[...,2,:]-a[...,0,:],dim=-1),dim=-1)
        nb=F.normalize(torch.linalg.cross(bm[...,1,:]-bm[...,0,:],bm[...,2,:]-bm[...,0,:],dim=-1),dim=-1)
        normals.append(average(1-(na*nb).sum(-1)))
        laps.append(huber(geometry.laplace(pp[:,:,q],side),geometry.laplace(tt[:,:,q],side)))
    invalid=(F.relu(1-p['curves'][:,:,0]).mean(1)+F.relu(p['curves'][:,:,0]-p['curves'][:,:,1]+1).mean(1))/100
    loss=(c['mesh_weight']*geom+c['phenotype_weight']*huber(p['ph']/scale,t['ph']/scale)+
          c['curve_weight']*huber(p['curves']/30,t['curves']/30)+c['relative_curve_weight']*huber(pc,tc,.05)+
          c['es_neighborhood_weight']*es+c['normal_weight']*torch.stack(normals).mean(0)+
          c['laplacian_weight']*torch.stack(laps).mean(0)+c['motion_weight']*huber((pred-pred[row,t['ed']][:,None])/5,(truth-truth[row,t['ed']][:,None])/5)+.1*invalid)
    return loss,dict(mesh_mean_mm=average(error),selection_score=average((p['ph'][:,[0,1,3,4]]-t['ph'][:,[0,1,3,4]]).abs()/scale[[0,1,3,4]])+.25*geom,invalid_fraction=p['invalid'].float(),EF_MAE_pp=(p['ph'][:,3]-t['ph'][:,3]).abs()),p,t

def relational_vector(p,t):
    n=len(p)
    if n<2:return p.sum(1)*0
    a,b=torch.pdist(p),torch.pdist(t.detach());valid=b>1e-6
    if not valid.any():return p.sum(1)*0
    vals=F.smooth_l1_loss(a[valid]/a[valid].mean().clamp_min(1e-6),b[valid]/b[valid].mean(),reduction='none')
    ii=torch.triu_indices(n,n,1,device=p.device)[:,valid]
    total=p.new_zeros(n).index_add(0,ii[0],vals).index_add(0,ii[1],vals)
    return total*(n/(2*len(vals)))

class TrainingSystem(nn.Module):
    def __init__(self,m,dec,geo,c,bank):
        super().__init__();self.student=m;self.decoder=dec;self.geometry=geo;self.c=c;self.bank=bank
    def train(self,mode=True):
        super().train(mode);self.decoder.eval();self.student.prior.eval();return self
    def forward(self,b):
        m=self.student;c=self.c
        with torch.autocast('cuda',dtype=torch.bfloat16):o=m(**args_of(b))
        future=m.propagate(o,b['delta_years']);z,res,prior=future['z1'],future['residual1'],future['prior1']
        assert torch.equal(b['is_async'],b['delta_years']>0)
        target=b['latent_std']-prior.detach();wp,wt=m.whiten(res),m.whiten(target)
        physical_loss,stats,_,_=physical_vector(z,b,self.decoder,self.geometry,c)
        cos=sum(1-F.cosine_similarity(res[:,ids],target[:,ids],dim=1,eps=.1) for ids in [m.shape_ids,m.motion_ids])/2
        nll=.5*average((wp-wt).square()/o['variance0']+o['variance0'].log())
        vec=physical_loss+c['latent_weight']*huber(wp,wt)+c['cosine_weight']*cos+c['relational_weight']*relational_vector(wp,wt)+c['uncertainty_weight']*nll
        sync=~b['is_async']
        # Keep original synchronous-only auxiliary terms, now patient-reduced.
        bp=F.normalize(self.bank['white'],dim=1,eps=.1);bd=self.bank['demo']
        pp=F.normalize(wp,dim=1,eps=.1);tp=F.normalize(wt.detach(),dim=1,eps=.1);demo=b['demo_cfp']
        valid=((demo[:,1,None]==bd[None,:,1]) & ((demo[:,0,None]-bd[None,:,0]).abs()<=c['negative_age_window_years']) & torch.isfinite(demo[:,2,None]) & torch.isfinite(bd[None,:,2]) & ((demo[:,2,None]-bd[None,:,2]).abs()<=c['negative_bsa_window_m2']))
        valid &= torch.arange(len(bp),device=pp.device)[None]!=b['row_index'][:,None]
        valid &= (tp@bp.T)<c['negative_max_cosine'];ok=valid.any(1)&sync
        contrast=wp.sum(1)*0
        if ok.any():
            logits=torch.cat([(pp*tp).sum(1,keepdim=True),(pp@bp.T).masked_fill(~valid,-torch.inf)],1)[ok]/c['contrastive_temperature']
            contrast=contrast.index_add(0,ok.nonzero().flatten(),F.cross_entropy(logits,torch.zeros(int(ok.sum()),device=pp.device,dtype=torch.long),reduction='none'))
        mask=b['vessel_mask'];count=mask.sum(1);mean=(b['vessels']*mask).sum(1)/count.clamp_min(1);vok=count>0
        aux=(F.smooth_l1_loss(o['vascular_prediction'],mean,reduction='none')*vok).sum(1)/vok.sum(1).clamp_min(1)
        vec=vec+c['contrastive_weight']*contrast+c['vascular_aux_weight']*aux*sync+c['transport_regularization']*average(future['change'].square())*b['is_async']
        w=weights(b['delta_years']);loss=(vec*w).mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite joint loss')
        return loss,{k:v.detach().mean() for k,v in stats.items()}|{'weight_mean':w.detach().mean(),'unweighted_loss':vec.detach().mean()}

from physical_chunks import vector_chunks
def physical_vector(z,b,decoder,geometry,c):
    return vector_chunks(_physical_vector_full,z,b,decoder,geometry,c)
