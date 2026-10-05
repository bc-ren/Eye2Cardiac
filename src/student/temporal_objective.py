"""Unchanged sync objective; patient-weighted async objective with exact scale."""
import torch
from torch.nn import functional as F
from objectives import loss_for as original_loss_for
from vector_reference import physical_vector,relational_vector,average,huber
from shared import S

def temporal_weights(dt):
    assert dt.ndim==1 and torch.isfinite(dt).all() and (dt>0).all()
    return S['async_coefficient']*torch.exp2(-dt/S['time_half_life_years'])

def async_vector(model,out,b,decoder,geometry,c):
    future=model.propagate(out,b['delta_years'])
    z,res,prior=future['z1'],future['residual1'],future['prior1']
    target=b['latent_std']-prior.detach()
    wp,wt=model.whiten(res),model.whiten(target)
    phys,stats,p,t=physical_vector(z,b,decoder,geometry,c)
    cosine=sum(1-F.cosine_similarity(res[:,ids],target[:,ids],dim=1,eps=.1)
        for ids in [model.shape_ids,model.motion_ids])/2
    nll=.5*average((wp-wt).square()/out['variance0']+out['variance0'].log())
    vec=(phys+c['latent_weight']*huber(wp,wt)+c['cosine_weight']*cosine+
        c['relational_weight']*relational_vector(wp,wt)+c['uncertainty_weight']*nll+
        c['transport_regularization']*average(future['change'].square()))
    assert vec.shape==b['delta_years'].shape and torch.isfinite(vec).all()
    return vec,stats,p,t

def loss_for(model,out,b,decoder,geometry,c,bank=None,asynchronous=False,anchor=None):
    if not asynchronous:
        return original_loss_for(model,out,b,decoder,geometry,c,bank,False,anchor)
    vec,stats,p,t=async_vector(model,out,b,decoder,geometry,c)
    w=temporal_weights(b['delta_years'])
    loss=(w*vec).mean()
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite time-attenuated loss')
    return loss,{k:v.detach().mean() for k,v in stats.items()}|{
        'loss':loss.detach(),'weight_mean':w.detach().mean(),'unweighted_loss':vec.detach().mean()},p,t
