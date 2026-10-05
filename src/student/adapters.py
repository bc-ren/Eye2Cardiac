"""S6-style selective scan; pure PyTorch, not the fused official VMamba kernel.

h[t] = exp(delta[t]*A) * h[t-1] + delta[t]*B[t]*u[t]
y[t] = sum(C[t]*h[t]) + D*u[t]. Float32 recurrence under AMP.
Parallel doubling is O(L log L) work, unlike the fused linear-work CUDA scan.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def recurrence(a,b,serial=False):
    if serial:
        h=torch.zeros_like(b[:,0]);out=[]
        for t in range(b.shape[1]):h=a[:,t]*h+b[:,t];out.append(h)
        return torch.stack(out,1)
    step=1
    while step<b.shape[1]:
        b=torch.cat([b[:,:step],b[:,step:]+a[:,step:]*b[:,:-step]],1)
        a=torch.cat([a[:,:step],a[:,step:]*a[:,:-step]],1)
        step*=2
    return b

class SelectiveScan(nn.Module):
    def __init__(self,d,n):
        super().__init__();self.d=d;self.n=n
        self.input=nn.Linear(d,2*d);self.conv=nn.Conv1d(d,d,3,padding=2,groups=d)
        self.params=nn.Linear(d,d+2*n);self.output=nn.Linear(d,d)
        self.log_a=nn.Parameter(torch.arange(1,n+1).float().log().repeat(d,1));self.skip=nn.Parameter(torch.ones(d))
        with torch.no_grad():self.params.bias[:d].fill_(-3)
    def forward(self,x,serial=False):
        u,gate=self.input(x).chunk(2,-1);u=F.silu(self.conv(u.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        raw,B,C=torch.split(self.params(u),[self.d,self.n,self.n],-1)
        delta=F.softplus(raw.float());A=-self.log_a.float().exp()
        a=torch.exp(delta[...,None]*A);b=delta[...,None]*B.float()[:,:,None,:]*u.float()[...,None]
        h=recurrence(a,b,serial);y=(h*C.float()[:,:,None,:]).sum(-1)+u.float()*self.skip
        return self.output((y*F.silu(gate.float())).to(x.dtype))

class Scan2D(nn.Module):
    def __init__(self,d,n):super().__init__();self.scans=nn.ModuleList([SelectiveScan(d,n) for _ in range(4)]);self.norm=nn.LayerNorm(d)
    def forward(self,x):
        b,h,w,d=x.shape;r=x.reshape(b,h*w,d);c=x.transpose(1,2).reshape(b,h*w,d)
        a=self.scans[0](r)+self.scans[1](r.flip(1)).flip(1)
        z=self.scans[2](c)+self.scans[3](c.flip(1)).flip(1)
        return self.norm((a.reshape(b,h,w,d)+z.reshape(b,w,h,d).transpose(1,2))/4)

class PolarScan(nn.Module):
    def __init__(self,d,n):
        super().__init__();self.radial=SelectiveScan(d,n);self.angular=SelectiveScan(d,n);self.norm=nn.LayerNorm(d)
    def forward(self,x,disc):
        b,h,w,d=x.shape;nr,nt=14,28
        radius=torch.linspace(0,2.83,nr,device=x.device);angle=torch.arange(nt,device=x.device)*2*math.pi/nt
        g=torch.stack([radius[:,None]*angle.cos(),radius[:,None]*angle.sin()],-1)[None]+disc[:,:2,None,None].permute(0,2,3,1)
        # Missing or outside anatomy contributes zero; caller applies explicit quality gate.
        pol=F.grid_sample(x.permute(0,3,1,2).float(),g.float(),align_corners=True,padding_mode='zeros').permute(0,2,3,1).to(x.dtype)
        radial=pol.transpose(1,2).reshape(b*nt,nr,d)
        radial=(self.radial(radial)+self.radial(radial.flip(1)).flip(1))/2
        radial=radial.reshape(b,nt,nr,d).transpose(1,2)
        ang=pol.reshape(b*nr,nt,d)
        # Circular padding of an entire revolution suppresses the arbitrary 0-degree seam.
        ext=torch.cat([ang,ang,ang],1)
        angular=(self.angular(ext)[:,nt:2*nt]+self.angular(ext.flip(1)).flip(1)[:,nt:2*nt])/2
        y=self.norm((radial+angular.reshape(b,nr,nt,d))/2)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij')
        dx=xx[None]-disc[:,0,None,None];dy=yy[None]-disc[:,1,None,None]
        rr=(dx.square()+dy.square()).sqrt()/2.83
        theta=torch.remainder(torch.atan2(dy,dx),2*math.pi)/(2*math.pi)
        # Append first angular column for continuous interpolation across the seam.
        y=torch.cat([y,y[:,:,:1]],2)
        inv=torch.stack([2*theta-1,2*rr-1],-1)
        return F.grid_sample(y.permute(0,3,1,2).float(),inv.float(),align_corners=True).permute(0,2,3,1).to(x.dtype)

class VesselScan(nn.Module):
    def __init__(self,d,n):
        super().__init__();self.scan=SelectiveScan(d,n);self.geometry=nn.Linear(4,d)
        self.message=nn.Linear(d,d);self.norm=nn.LayerNorm(d)
    def forward(self,x,paths,mask,geometry,adjacency):
        b,h,w,d=x.shape;k,l=paths.shape[1:3]
        p=F.grid_sample(x.permute(0,3,1,2).float(),paths.float(),align_corners=True).permute(0,2,3,1).to(x.dtype)
        seq=p.reshape(b*k,l,d);seq=(self.scan(seq)+self.scan(seq.flip(1)).flip(1))/2
        nodes=seq.mean(1).reshape(b,k,d)+self.geometry(geometry.to(x.dtype))
        a=adjacency.to(x.dtype)*mask[:,None,:]*mask[:,:,None];a=a/a.sum(-1,keepdim=True).clamp_min(1)
        nodes=self.norm(nodes+self.message(a@nodes))*mask[...,None]
        return nodes.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

class Adapter(nn.Module):
    def __init__(self,arm,d=64,n=8,gamma=.001):
        super().__init__();self.arm=arm;self.ln=nn.LayerNorm(1024);self.down=nn.Linear(1024,d);self.up=nn.Linear(d,1024);self.gamma=nn.Parameter(torch.tensor(gamma))
        if arm=='mlp':self.core=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        elif arm=='attention':self.core=nn.TransformerEncoderLayer(d,4,4*d,dropout=.1,batch_first=True,norm_first=True)
        else:self.core=Scan2D(d,n)
        self.polar=PolarScan(d,n) if arm in ['polar','cartesian_polar','full','full_ssl'] else None
        self.vessel=VesselScan(d,n) if arm in ['cartesian_vessel','full','full_ssl'] else None
        self.polar_gate=nn.Parameter(torch.tensor(-1.));self.graph_gamma=nn.Parameter(torch.tensor(.001))
    def forward(self,tokens,disc,paths,segment_mask,segment_geometry,adjacency):
        x=self.down(self.ln(tokens));b,l,d=x.shape;assert l==196
        grid=x.reshape(b,14,14,d)
        base=self.core(x) if self.arm in ['mlp','attention'] else self.core(grid).flatten(1,2)
        if self.polar is not None:
            p=self.polar(grid,disc).flatten(1,2);valid=disc[:,2,None,None].to(base.dtype)
            gate=valid if self.arm=='polar' else valid*self.polar_gate.sigmoid()
            base=(1-gate)*base+gate*p
        if self.vessel is not None:
            graph=self.vessel(grid,paths,segment_mask,segment_geometry,adjacency)
            base=base+self.graph_gamma*graph[:,None]
        return tokens+self.gamma*self.up(base)

class PatchQueries(nn.Module):
    def __init__(self,arm,c):
        super().__init__();self.adapter=Adapter(arm,c['bottleneck'],c['state_dim'],c['gamma_init'])
        self.proj=nn.Linear(1024,256);self.seeds=nn.Parameter(torch.randn(1,4,256)*.02)
        self.attention=nn.MultiheadAttention(256,4,batch_first=True);self.up=nn.Linear(256,1024)
    def forward(self,tokens,**anatomy):
        adapted=self.adapter(tokens,**anatomy);x=self.proj(adapted);q,_=self.attention(self.seeds.expand(len(x),-1,-1),x,x,need_weights=False)
        return self.up(q+self.seeds),adapted
