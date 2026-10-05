"""Fold-fitted preprocessing, 24D branch projections and exact full-batch Cox."""
import random
import numpy as np,torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_curve
from runlib import *
FORMULA=module('framingham_formula',HERE/'clinical_formula.py')
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False

class ClinicalScore:
 def fit(self,x):
  self.fill=np.nanmedian(x,axis=0)
  for j in (1,6,7):
   v=x[np.isfinite(x[:,j]),j];assert len(v);self.fill[j]=float(v.mean()>=.5)
  assert np.isfinite(self.fill).all();return self
 def transform(self,x):
  q=np.where(np.isfinite(x),x,self.fill)
  return FORMULA.framingham_chd(q[:,0],q[:,1]<.5,q[:,2]*38.67,q[:,3]*38.67,q[:,4],q[:,5],q[:,6],q[:,7]).astype(np.float32)[:,None]

class Scale:
 def fit(self,x):
  self.median=np.nanmedian(x,axis=0);assert np.isfinite(self.median).all()
  q=np.where(np.isfinite(x),x,self.median);self.mean=q.mean(0);self.std=q.std(0);self.std=np.where(self.std>1e-8,self.std,1);return self
 def transform(self,x):return ((np.where(np.isfinite(x),x,self.median)-self.mean)/self.std).astype(np.float32)

class Net(nn.Module):
 def __init__(self,dims,diagnosis=False,prevalence=.5,offset=False):
  super().__init__();self.dims=dims
  self.branches=nn.ModuleDict({k:nn.Sequential(nn.Linear(d,C['branch_hidden']),nn.GELU(),nn.Dropout(C['dropout']),nn.Linear(C['branch_hidden'],24),nn.GELU()) for k,d in dims.items()})
  self.head=nn.Linear(24*len(dims),1)
  if offset:nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
  elif diagnosis:nn.init.constant_(self.head.bias,float(np.log(prevalence/(1-prevalence))))
 def forward(self,x,offset=None):
  cat=torch.cat([self.branches[k](x[k]) for k in self.dims],dim=1)
  assert cat.shape[1]==24*len(self.dims)
  r=self.head(cat).squeeze(1)
  return r if offset is None else r+offset

def cox_loss(risk,time,event):
 # Exact Breslow ties: all patients at the event time remain in its risk set.
 order=torch.argsort(time,descending=True,stable=True);r=risk[order].double();t=time[order];e=event[order].double()
 _,counts=torch.unique_consecutive(t,return_counts=True);ends=counts.cumsum(0)-1
 logden=torch.repeat_interleave(torch.logcumsumexp(r,0)[ends],counts)
 assert e.sum()>0
 return (-(e*(r-logden)).sum()/e.sum()).float()

def fit_pre(features,tr,branches,raw_clinical):
 clin=ClinicalScore().fit(raw_clinical[tr]) if 'clinical' in branches else None
 arrays={k:clin.transform(raw_clinical) if k=='clinical' else features[k] for k in branches}
 scales={k:Scale().fit(arrays[k][tr]) for k in branches}
 return clin,scales

def matrix(features,indices,branches,clin,scales,raw_clinical,device='cuda'):
 return {k:torch.tensor(scales[k].transform(clin.transform(raw_clinical[indices]) if k=='clinical' else features[k][indices]),device=device) for k in branches}

def predict(bundle,features,indices,raw_clinical,offset=None):
 net=Net(bundle['dims']).cuda();net.load_state_dict(bundle['state']);net.eval();result=[]
 with torch.no_grad():
  for j in range(0,len(indices),4096):
   ix=indices[j:j+4096];x=matrix(features,ix,bundle['branches'],bundle['clinical'],bundle['scales'],raw_clinical)
   off=None if offset is None else torch.tensor(offset[j:j+4096],device='cuda',dtype=torch.float32)
   result.append(net(x,off).cpu().numpy())
 return np.concatenate(result)

def train(features,raw_clinical,tr,va,y,time,branches,diagnosis,seed,metric,offset_train=None,offset_val=None,fixed_epochs=None):
 seed_all(seed);clinical,scales=fit_pre(features,tr,branches,raw_clinical)
 x=matrix(features,tr,branches,clinical,scales,raw_clinical)
 xv=matrix(features,va,branches,clinical,scales,raw_clinical) if va is not None else None
 net=Net({k:x[k].shape[1] for k in branches},diagnosis,float(y[tr].mean()),offset_train is not None).cuda()
 opt=torch.optim.AdamW(net.parameters(),lr=C['learning_rate'],weight_decay=C['weight_decay']);scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=C['lr_decay_per_epoch'])
 target=torch.tensor(y[tr],device='cuda',dtype=torch.float32)
 ts=None if diagnosis else torch.tensor(time[tr],device='cuda',dtype=torch.float64)
 ot=None if offset_train is None else torch.tensor(offset_train,device='cuda',dtype=torch.float32)
 ov=None if offset_val is None else torch.tensor(offset_val,device='cuda',dtype=torch.float32)
 best=-float('inf');best_epoch=0;best_state=None;history=[];best_pred=None
 for epoch in range(1,(fixed_epochs or C['max_epochs'])+1):
  net.train();opt.zero_grad(set_to_none=True);r=net(x,ot)
  loss=nn.functional.binary_cross_entropy_with_logits(r,target) if diagnosis else cox_loss(r,ts,target)
  assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
  row=dict(epoch=epoch,loss=float(loss.detach()),lr=opt.param_groups[0]['lr'],grad_norm=float(grad));scheduler.step()
  if va is not None:
   net.eval()
   with torch.no_grad():p=net(xv,ov).cpu().numpy()
   value=float(average_precision_score(y[va],p)) if diagnosis else metric(time[va],y[va],p)
   row['selection_metric']=value
   if value>best:
    best=value;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};best_pred=p.copy()
  else:best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
  history.append(row)
  if va is not None and epoch-best_epoch>=C['patience']:break
 assert best_state is not None
 bundle=dict(state=best_state,dims=net.dims,branches=branches,clinical=clinical,scales=scales,epoch=best_epoch,seed=seed,config=C,fields11=FIELDS11,train_n=len(tr),offset=offset_train is not None)
 return bundle,best_pred,history

def threshold(y,score):
 assert np.isfinite(score).all() and len(np.unique(y))==2
 fpr,tpr,t=roc_curve(y,score);ok=np.isfinite(t);return float(t[ok][np.argmax((tpr-fpr)[ok])])

def fit_baseline(r,time,event):
 shift=float(np.max(r));weights=np.exp(r-shift);times=np.unique(time[event.astype(bool)])
 increment=np.array([np.sum(event[time==t])/weights[time>=t].sum() for t in times])
 return dict(times=times,hazard=np.cumsum(increment),shift=shift,max_followup=float(np.max(time)))
def risks(r,baseline):
 ix=np.searchsorted(baseline['times'],C['horizons'],side='right')-1
 h=np.where(ix>=0,baseline['hazard'][np.maximum(ix,0)],0.)
 return -np.expm1(-np.exp(np.asarray(r)-baseline['shift'])[:,None]*h[None,:])
