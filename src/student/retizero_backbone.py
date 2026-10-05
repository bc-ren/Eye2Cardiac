"""Original RetiZero, with only Transformer blocks 20--23 trainable."""
import hashlib
import sys
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from shared import S

TRAINABLE_BLOCKS = (20, 21, 22, 23)


def is_trainable_name(name):
    return any(name.startswith(f'blocks.{i}.') for i in TRAINABLE_BLOCKS)


def frozen_hash(vit):
    h = hashlib.sha256()
    for name, value in vit.state_dict().items():
        if not is_trainable_name(name):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_scope(backbone):
    params = dict(backbone.vit.named_parameters())
    assert len(backbone.vit.blocks) == 24
    assert all(p.requires_grad == is_trainable_name(k) for k, p in params.items())
    assert any(p.requires_grad for p in params.values())
    return dict(trainable_blocks_1based=[21, 22, 23, 24],
                trainable_parameters=sum(p.numel() for p in params.values() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in params.values() if not p.requires_grad))


class RetiZeroLast4(nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0, S['retina_source'])
        from clip_modules.modeling.LoraRETFound import lora
        original = lora(pretrained=False, R=8)
        state = torch.load(S['retina_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        state = state.get('model', state.get('state_dict', state))
        prefix = 'vision_model.model.'
        visual = {k.removeprefix('module.')[len(prefix):]: v for k, v in state.items()
                  if k.removeprefix('module.').startswith(prefix)}
        assert visual, 'No original RetiZero vision weights found'
        original.load_state_dict(visual, strict=True)
        self.vit = original.lora_vit
        # Keep the original LoRA parameterization; do not merge/reinitialize it.
        self.vit.requires_grad_(False)
        for i in TRAINABLE_BLOCKS:
            self.vit.blocks[i].requires_grad_(True)
        self.chunk = S['retina_micro_batch']
        assert_scope(self)

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        for i in TRAINABLE_BLOCKS:
            self.vit.blocks[i].train(mode)
        return self

    def tokens(self, x):
        with torch.no_grad():
            x = self.vit.patch_embed.proj(x)
            x = self.vit.patch_embed.norm(x.flatten(2).transpose(1, 2))
            x = self.vit.pos_drop(torch.cat([self.vit.cls_token.expand(len(x), -1, -1), x], 1) + self.vit.pos_embed)
            for block in self.vit.blocks[:20]:
                x = block(x)
        for block in self.vit.blocks[20:]:
            x = block(x)
        # Frozen final norm still transmits gradients to the last four blocks.
        return self.vit.norm(x)

    def forward(self, images):
        assert images.shape[1:] == (3, 224, 224)
        parts = []
        for x in images.split(self.chunk):
            y = checkpoint(self.tokens, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else self.tokens(x)
            assert y.shape[1:] == (197, 1024)
            parts.append(y)
        return torch.cat(parts)
