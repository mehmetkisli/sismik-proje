# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Kaynak: https://github.com/shenghanlin/SeismicFoundationModel/blob/main/SFM-Finetune/models_Segmentation.py
# (yerel kopya — degisiklik yapilmadi)
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
import numpy as np


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        self.decoder = VIT_MLAHead(mla_channels=self.embed_dim, num_classes=self.num_classes)
        self.segmentation_head = SegmentationHead(
            in_channels=16, out_channels=self.num_classes, kernel_size=3,
        )
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        _H, _W = H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        featureskip = []
        featureskipnum = 1
        for blk in self.blocks:
            x = blk(x)
            if featureskipnum % (len(self.blocks) // 4) == 0:
                featureskip.append(x[:, 1:, :])
            featureskipnum += 1
        x = self.decoder(featureskip[0], featureskip[1], featureskip[2], featureskip[3], h=_H, w=_W)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super().__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = F.interpolate(self.head2(mla_p2),
                              (4 * mla_p2.shape[-2], 4 * mla_p2.shape[-1]),
                              mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(mla_p3),
                              (4 * mla_p3.shape[-2], 4 * mla_p3.shape[-1]),
                              mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(mla_p4),
                              (4 * mla_p4.shape[-2], 4 * mla_p4.shape[-1]),
                              mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(mla_p5),
                              (4 * mla_p5.shape[-2], 4 * mla_p5.shape[-1]),
                              mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)


class VIT_MLAHead(nn.Module):
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, num_classes=6,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_classes = num_classes
        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls = nn.Conv2d(4 * self.mlahead_channels, self.num_classes, 3, padding=1)

    def forward(self, x1, x2, x3, x4, h=14, w=14):
        B, n_patch, hidden = x1.size()
        if h == w:
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x1 = x1.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x2 = x2.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x3 = x3.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x4 = x4.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.mlahead(x1, x2, x3, x4)
        x = self.cls(x)
        x = F.interpolate(x, size=(h * 16, w * 16), mode='bilinear', align_corners=True)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
