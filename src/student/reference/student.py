"""RetiCardiac base: retinal adaptation, CardiacAE prior and time transport."""
import torch
from torch import nn
from shared import S
from model import CardiacQueryStudent,CardiacDecoder,scale_gradient
from adapters import PatchQueries

ANATOMY=['disc','paths','segment_mask','segment_geometry','adjacency']
def args_of(b):return {k:b[k] for k in ['images','cls','eye_mask','vessels','vessel_mask','demo_cfp',*ANATOMY]}

class RetiCardiacBase(CardiacQueryStudent):
    def __init__(self,c,prior,scalers,white,arm):
        super().__init__(c,prior,scalers,white,backbone=nn.Identity())
        self.patch_encoder=PatchQueries(arm,S);self.arm=arm
        self.repair_variant=c.get('repair_variant','legacy')
        assert self.repair_variant in S['repair_variants']
        if self.repair_variant in ['gate_init','combined']:
            with torch.no_grad():
                self.patch_encoder.adapter.gamma.fill_(S['repair_gate_init'])
                self.patch_encoder.adapter.graph_gamma.fill_(S['repair_graph_init'])
        if self.repair_variant in ['head_init','combined']:
            for head in [self.shape_head,self.motion_head]:
                nn.init.normal_(head[-1].weight,std=S['repair_head_std'])
        self.direct_readout=None
        if self.repair_variant in ['direct_readout','combined']:
            # Short path from adapted spatial features to teacher-native residual.
            self.direct_readout=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,96))
            nn.init.normal_(self.direct_readout[-1].weight,std=S['repair_readout_std'])
            nn.init.zeros_(self.direct_readout[-1].bias)
    def forward(self,images,cls,eye_mask,vessels,vessel_mask,demo_cfp,disc,paths,segment_mask,segment_geometry,adjacency,backbone_gradient=1.):
        b=len(images);assert images.shape[1:]==(2,196,1024)
        assert eye_mask.any(1).all() and eye_mask.dtype==torch.bool
        valid=eye_mask.flatten();anatomy={k:v.flatten(0,1)[valid] for k,v in dict(disc=disc,paths=paths,segment_mask=segment_mask,segment_geometry=segment_geometry,adjacency=adjacency).items()}
        query,adapted=self.patch_encoder(images.flatten(0,1)[valid],**anatomy)
        qe=query.new_zeros((2*b,4,1024));qe[valid]=query;qe=qe.reshape(b,2,4,1024)
        qe=self.query_project(qe)+self.eye_embeddings;keys=qe.flatten(1,2)
        shared,att=self.cross_eye(self.shared_seeds.expand(b,-1,-1),keys,keys,key_padding_mask=(~eye_mask).repeat_interleave(4,1),need_weights=True)
        shared=self.shared_norm(shared+self.shared_seeds);ew=eye_mask.float()/eye_mask.sum(1,keepdim=True)
        v=self.vascular(torch.cat([torch.where(vessel_mask,vessels,0),vessel_mask.float()],-1));v=(v*ew[...,None]).sum(1)
        cardiac=self.cardio_fusion(torch.cat([shared[:,:3],v[:,None].expand(-1,3,-1)],-1))
        shape=self.shape_head(torch.cat([cardiac[:,0],cardiac[:,1]],-1));motion=self.motion_head(torch.cat([cardiac[:,2],cardiac[:,0]],-1))
        r0=torch.cat([shape[:,:32],motion[:,:32],shape[:,32:],motion[:,32:]],1).float()
        if self.direct_readout is not None:
            pooled=adapted.mean(1)
            bilateral=pooled.new_zeros((b*2,1024));bilateral[valid]=pooled
            pooled=(bilateral.reshape(b,2,1024)*ew[...,None]).sum(1)
            r0=r0+self.direct_readout(pooled).float()
        r0=scale_gradient(r0,backbone_gradient);prior0=self.prior(demo_cfp)
        var=.05+3.95*torch.sigmoid(self.variance(cardiac.flatten(1)).float())
        return dict(r0=r0,z0=prior0+r0,prior0=prior0,demo0=demo_cfp,retina=(cls*ew[...,None]).sum(1).float(),
            shared_tokens=shared.float(),eye_specific=(qe-shared[:,None]).float(),attention=att.float(),
            vascular_prediction=self.vascular_aux(shared[:,3]).float(),variance0=scale_gradient(var,backbone_gradient),
            adapted_semantics=adapted.float().mean(1),original_semantics=images.flatten(0,1)[valid].float().mean(1))
    def export(self):return {k:v.detach().cpu() for k,v in self.state_dict().items()}
    def restore(self,s):self.load_state_dict(s,strict=True)

# Historical internal import compatibility; the public model is RetiCardiac.
Student = RetiCardiacBase

def make_models(c,arm):
    from engine import artifacts
    from objectives import Geometry
    p,sc,w=artifacts(S['fold'])
    return RetiCardiacBase(c,p,sc,w,arm).cuda(),CardiacDecoder(c,sc).cuda(),Geometry(c).cuda()
