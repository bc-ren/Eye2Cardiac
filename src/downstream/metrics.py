"""Patient-weighted metrics: pinned algorithm from the previous clinical evaluation."""
import numpy as np
from numba import njit
METRICS=['AUROC','AUPRC','Accuracy','Sensitivity','Specificity','PPV','NPV','F1','Brier','TP','FP','TN','FN']
@njit(cache=True)
def div(a,b):return a/b if b>0 else np.nan
@njit(cache=True)
def binary(y,p,order,weights,threshold):
    P=0.;N=0.;tp=0.;fp=0.;tn=0.;fn=0.;brier=0.
    for i in range(len(y)):
        w=weights[i];P+=w*y[i];N+=w*(1-y[i]);brier+=w*(p[i]-y[i])**2
        if p[i]>=threshold:tp+=w*y[i];fp+=w*(1-y[i])
        else:tn+=w*(1-y[i]);fn+=w*y[i]
    cp=0.;cn=0.;auc=0.;ap=0.;start=0
    while start<len(y):
        end=start+1
        while end<len(y) and p[order[end]]==p[order[start]]:end+=1
        dp=0.;dn=0.
        for j in range(start,end):
            i=order[j];dp+=weights[i]*y[i];dn+=weights[i]*(1-y[i])
        old=cp;cp+=dp;cn+=dn
        if P>0 and N>0:auc+=dn/N*(old+cp)/(2*P)
        if P>0 and cp+cn>0:ap+=dp/P*cp/(cp+cn)
        start=end
    return np.array([auc if P>0 and N>0 else np.nan,ap if P>0 else np.nan,div(tp+tn,P+N),div(tp,P),div(tn,N),
        div(tp,tp+fp),div(tn,tn+fn),div(2*tp,2*tp+fp+fn),div(brier,P+N),tp,fp,tn,fn])
@njit(cache=True)
def add(tree,i,v):
    i+=1
    while i<len(tree):tree[i]+=v;i+=i&-i
@njit(cache=True)
def prefixsum(tree,end):
    total=0.
    while end>0:total+=tree[end];end-=end&-end
    return total
@njit(cache=True)
def concordance_sorted(t,e,order,rank,lower,upper,weights,nrisk):
    tree=np.zeros(nrisk+1);total=0.;num=0.;den=0.;start=0
    while start<len(t):
        end=start+1
        while end<len(t) and t[order[end]]==t[order[start]]:end+=1
        for j in range(start,end):
            i=order[j]
            if not e[i]:add(tree,rank[i],weights[i]);total+=weights[i]
        for j in range(start,end):
            i=order[j]
            if e[i]:
                lo=prefixsum(tree,lower[i]);hi=prefixsum(tree,upper[i]);num+=weights[i]*(lo+.5*(hi-lo));den+=weights[i]*total
        for j in range(start,end):
            i=order[j]
            if e[i]:add(tree,rank[i],weights[i]);total+=weights[i]
        start=end
    return div(num,den)
def cargs(t,e,r):
    t=np.asarray(t,float);e=np.asarray(e,bool);r=np.asarray(r,float);v=np.unique(r)
    return t,e,np.argsort(-t,kind='stable'),np.searchsorted(v,r),np.searchsorted(v,r-1e-8,'left'),np.searchsorted(v,r+1e-8,'right'),len(v)
def cindex(args,w):return float(concordance_sorted(*args[:-1],w,args[-1]))
def resample_weights(y,rng):
    a=np.flatnonzero(y);b=np.flatnonzero(~y)
    return np.bincount(np.r_[rng.choice(a,len(a),replace=True),rng.choice(b,len(b),replace=True)],minlength=len(y)).astype(float)
def ci(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return (float(np.quantile(a,.025)),float(np.quantile(a,.975)),len(a)) if len(a) else (None,None,0)
