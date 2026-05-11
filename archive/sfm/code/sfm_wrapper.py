"""SFM wrapper: 5-kanal 2.5D girdiyi SFM ViT-Base/16'ya adapte eder.

Tasarim:
- nn.Conv2d(5, 3, 1) ile 5 -> 3 kanal ogrenilebilir indirgeme (kanal-mean ile init)
- SFM ViT-Base/16 encoder (pos_embed interpolation ile 384x384 destegi)
- VIT_MLAHead decoder (4 intermediate ViT block'tan feature aggregation)
- 6-sinif segmentation output

Kullanim:
    model = build_sfm_segmenter(num_classes=6, pretrained_path='sfm/pretrained/Base-512.pth',
                                input_size=384, in_channels=5, freeze_encoder=True)
"""
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models_segmentation import vit_base_patch16


def _interpolate_pos_embed(pos_embed, new_size, patch_size=16):
    """ViT pos embedding'i farkli resolution'a interpolate et.
    pos_embed shape: (1, 1 + N_old, embed_dim) — 1 cls token + N_old patch token
    new_size: yeni input image size (kare varsay)
    Donus: (1, 1 + N_new, embed_dim)
    """
    cls_pe = pos_embed[:, :1, :]
    patch_pe = pos_embed[:, 1:, :]
    n_old = patch_pe.shape[1]
    side_old = int(math.sqrt(n_old))
    assert side_old * side_old == n_old, f"pos_embed not square: {n_old}"
    side_new = new_size // patch_size
    if side_new == side_old:
        return pos_embed
    # 2D interpolation
    dim = patch_pe.shape[-1]
    patch_pe = patch_pe.reshape(1, side_old, side_old, dim).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(patch_pe, size=(side_new, side_new), mode='bicubic', align_corners=False)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, side_new * side_new, dim)
    return torch.cat([cls_pe, patch_pe], dim=1)


class SFMSegmenter(nn.Module):
    """5-kanal sismik girdi -> SFM ViT-Base -> 6-sinif segmentasyon."""

    def __init__(self, num_classes=6, input_size=384, in_channels=5):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        # SFM 1-channel grayscale ile pretrain — bizim 5-kanal 2.5D'yi 1-kanala indirge
        self.channel_adapter = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            # 5 kanali esit agirlikla ortala -> 1 kanal
            w = torch.full((1, in_channels, 1, 1), 1.0 / in_channels)
            self.channel_adapter.weight.copy_(w)
            self.channel_adapter.bias.zero_()

        # SFM ViT-Base/16, in_chans=1 (SFM pretrain konfigurasyonu)
        # Pretrained yuklenince pos_embed otomatik interpolate edilir
        self.vit = vit_base_patch16(num_classes=num_classes, img_size=input_size, in_chans=1)

    def load_sfm_pretrained(self, ckpt_path: Path):
        """SFM Base-512.pth'i yukle. Sadece encoder agirliklari (patch_embed + blocks + pos_embed).

        SFM checkpoint Linux'ta pickle edilmis, Windows'ta PosixPath'i unpickle edemez.
        Gecici monkey-patch ile cozuluyor.
        """
        import pathlib, sys
        if sys.platform == 'win32':
            _posix_backup = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            finally:
                pathlib.PosixPath = _posix_backup
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # SFM checkpoint formati: ya direkt state_dict ya da {'model': state_dict}
        if 'model' in ckpt:
            sd = ckpt['model']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt

        # Pos embed interpolation (eger native size != bizim input size)
        if 'pos_embed' in sd:
            old_pe = sd['pos_embed']
            new_pe = _interpolate_pos_embed(old_pe, new_size=self.input_size, patch_size=16)
            sd['pos_embed'] = new_pe
            print(f"  pos_embed interpolated: {old_pe.shape} -> {new_pe.shape}")

        # Decoder/head agirliklari ImageNet veya MAE ile gelmeyebilir; bunlari yuklemiyoruz
        keep_keys = set()
        for k in sd.keys():
            if k.startswith('decoder.') or k.startswith('segmentation_head.') or k.startswith('cls.') or k.startswith('head.'):
                continue
            if k.startswith('mask_token') or k.startswith('decoder_'):
                continue
            keep_keys.add(k)

        filtered_sd = {k: sd[k] for k in keep_keys}
        missing, unexpected = self.vit.load_state_dict(filtered_sd, strict=False)
        print(f"  SFM weights yuklendi: {len(filtered_sd)} key matched")
        print(f"  Missing (yeni decoder/head): {len(missing)} key")
        print(f"  Unexpected (SFM-only): {len(unexpected)} key")
        if unexpected:
            print(f"    ornek unexpected: {list(unexpected)[:5]}")
        return missing, unexpected

    def freeze_encoder(self):
        """ViT encoder'i (patch_embed, blocks, norm, pos_embed, cls_token) freeze et."""
        for name, p in self.vit.named_parameters():
            if name.startswith('decoder.') or name.startswith('segmentation_head.') or name.startswith('cls.'):
                p.requires_grad = True
            else:
                p.requires_grad = False
        # channel_adapter trainable
        for p in self.channel_adapter.parameters():
            p.requires_grad = True

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"  Encoder frozen: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M trainable")

    def forward(self, x):
        # x: (B, 5, H, W)
        x = self.channel_adapter(x)   # (B, 3, H, W)
        out = self.vit(x)             # (B, num_classes, H, W)
        return out


def build_sfm_segmenter(num_classes=6, pretrained_path=None, input_size=384, in_channels=5,
                       freeze_encoder=True):
    """Tum pipeline'i kuran factory."""
    model = SFMSegmenter(num_classes=num_classes, input_size=input_size,
                         in_channels=in_channels)
    if pretrained_path is not None and Path(pretrained_path).exists():
        print(f"SFM pretrained yukleniyor: {pretrained_path}")
        model.load_sfm_pretrained(Path(pretrained_path))
    else:
        print(f"UYARI: pretrained agirliklar yok ({pretrained_path}) - random init")
    if freeze_encoder:
        model.freeze_encoder()
    return model


if __name__ == '__main__':
    # Sanity check
    print("=== SFM Segmenter sanity test ===")
    model = build_sfm_segmenter(num_classes=6, pretrained_path=None,
                                input_size=384, in_channels=5, freeze_encoder=True)
    dummy = torch.randn(2, 5, 384, 384)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}  (beklenen: [2, 6, 384, 384])")
    print(f"Assert OK: {tuple(out.shape) == (2, 6, 384, 384)}")
