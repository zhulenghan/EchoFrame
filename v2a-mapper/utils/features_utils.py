from typing import Literal, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize



def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        enable_conditions: bool = True,
    ):
        super().__init__()

        if enable_conditions:
            self.clip_model = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
                                                           return_transform=False)
            self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                             std=[0.26862954, 0.26130258, 0.27577711])
            self.clip_model = patch_clip(self.clip_model)

        else:
            self.clip_model = None


    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_video_with_clip(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224  # 384
        x = self.clip_preprocess(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            outputs.append(self.clip_model.encode_image(x[i:i + batch_size], normalize=True))
        x = torch.cat(outputs, dim=0)
        # x = self.clip_model.encode_image(x, normalize=True)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
