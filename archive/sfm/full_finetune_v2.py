"""SFM full fine-tune: encoder unfrozen + layer-wise LR decay + Lovasz FP32.

v9 pipeline'inin SFM'ye adapte edilmis hali:
- SFM ViT-Base/16 encoder (UNFROZEN, layer-wise LR decay)
- 5->1 channel adapter (SFM 1-channel pretrain ile uyumlu)
- VIT_MLAHead decoder (yeni, random init)
- QuadrupleLoss (LS-CE + Dice + Focal + Lovasz), Lovasz FP32 wrap
- Mixup, xline-aware augmentation
- Multi-scale TTA (HFlip + polarity + scale 0.75/1.0/1.25)
- AMP, gradient accumulation

Cikti: results/metrics/sfm_finetune_metrics.json + checkpoint
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent / "code"))
from sfm_wrapper import build_sfm_segmenter

# ─── Konfig ─────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
CKPT_DIR    = PROJECT_DIR / "checkpoints_v7"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PRETRAINED  = PROJECT_DIR / "sfm" / "pretrained" / "SFM-Base-512.pth"
CHECKPOINT_PATH = CKPT_DIR / "sfm_finetune_v2_checkpoint.pth"
BEST_MODEL_PATH = CKPT_DIR / "sfm_finetune_v2_best.pth"

# v2 degisiklikleri:
# - BASE_LR_ENC 5e-5 -> 2e-4 (4x artis, encoder daha aktif fine-tune)
# - LAYER_DECAY 0.75 -> 0.9 (deeper layers de ogrenir)
# - PATIENCE 15 -> 20, NUM_EPOCHS 60 -> 80
# - RARE_BOOST 10 -> 15 (Class 4+5 birlikte daha cok ornek)
SEED         = 42
IMG_SIZE     = (384, 384)
N_NEIGHBORS  = 2
N_CHANNELS   = 2 * N_NEIGHBORS + 1
NUM_CLASSES  = 6
BATCH_SIZE   = 2
ACCUM_STEPS  = 12
NUM_EPOCHS   = 80
PATIENCE     = 20
WARMUP_EPOCHS = 5
BASE_LR_ENC  = 2e-4
LR_HEAD      = 1e-3
WEIGHT_DECAY = 0.05
LAYER_DECAY  = 0.9
MIXUP_ALPHA  = 0.2
LABEL_SMOOTH = 0.1

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Cihaz: {device}  AMP: {use_amp}")


# ─── Veri yukle ─────────────────────────────────────────────────────────────
print("\nVeri yukleniyor...")
train_seismic = np.load(DATA_DIR / "train" / "train_seismic.npy")
train_labels  = np.load(DATA_DIR / "train" / "train_labels.npy")
test1_seismic = np.load(DATA_DIR / "test_once" / "test1_seismic.npy")
test1_labels  = np.load(DATA_DIR / "test_once" / "test1_labels.npy")
test2_seismic = np.load(DATA_DIR / "test_once" / "test2_seismic.npy")
test2_labels  = np.load(DATA_DIR / "test_once" / "test2_labels.npy")

n_inlines = train_seismic.shape[0]; n_xlines = train_seismic.shape[1]
BUFFER = 2
val_start, val_end = 160, 240
val_inline_idx   = np.arange(val_start, val_end)
train_inline_idx = np.concatenate([np.arange(0, val_start - BUFFER),
                                   np.arange(val_end + BUFFER, n_inlines)])
train_xline_idx  = np.arange(0, n_xlines)
train_inline_mask = np.zeros(n_inlines, dtype=bool)
train_inline_mask[train_inline_idx] = True

train_mean = train_seismic.mean(); train_std = train_seismic.std() + 1e-8
train_norm = ((train_seismic - train_mean) / train_std).astype(np.float32)
test1_norm = ((test1_seismic - train_mean) / train_std).astype(np.float32)
test2_norm = ((test2_seismic - train_mean) / train_std).astype(np.float32)
print(f"  Train inline: {len(train_inline_idx)}, val: {len(val_inline_idx)}, xline: {len(train_xline_idx)}")


# ─── Augmentation ───────────────────────────────────────────────────────────
inline_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, border_mode=0, p=0.5),
    A.ElasticTransform(alpha=80, sigma=10, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.GaussNoise(std_range=(0.001, 0.015), p=0.3),
    A.CoarseDropout(max_holes=6, max_height=20, max_width=20, fill=0, p=0.2),
])
# xline-aware: Class 4 morfolojisi icin agresif elastic+grid
xline_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.20, rotate_limit=12, border_mode=0, p=0.6),
    A.ElasticTransform(alpha=140, sigma=12, p=0.55),
    A.GridDistortion(num_steps=6, distort_limit=0.30, p=0.50),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.GaussNoise(std_range=(0.001, 0.015), p=0.3),
    A.CoarseDropout(max_holes=6, max_height=20, max_width=20, fill=0, p=0.2),
])


# ─── Dataset ────────────────────────────────────────────────────────────────
class F3Dataset25D(Dataset):
    def __init__(self, volume, labels, indices, axis=0, img_size=IMG_SIZE, transform=None,
                 augment_polarity=False, train_inline_mask=None, n_neighbors=N_NEIGHBORS):
        self.volume = volume; self.labels = labels; self.indices = indices
        self.axis = axis; self.n_slices = volume.shape[axis]
        self.img_size = img_size; self.transform = transform
        self.augment_polarity = augment_polarity; self.n_neighbors = n_neighbors
        self.train_inline_mask = train_inline_mask

    def _slice(self, vol, idx):
        if self.axis == 0: return vol[idx]
        s = vol[:, idx]
        if self.train_inline_mask is not None: s = s[self.train_inline_mask]
        return s

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        slices = []
        for k in range(-self.n_neighbors, self.n_neighbors + 1):
            j = max(0, min(self.n_slices - 1, i + k))
            slices.append(self._slice(self.volume, j).astype(np.float32))
        img  = np.stack(slices, axis=-1)
        mask = self._slice(self.labels, i).astype(np.int64)
        if self.augment_polarity and random.random() > 0.5:
            img = -img
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask.astype(np.uint8))
            img = aug["image"]; mask = aug["mask"].astype(np.int64)
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask.copy())
        img_t = F.interpolate(img_t.unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=False).squeeze(0)
        mask_t = F.interpolate(mask_t.float().unsqueeze(0).unsqueeze(0),
                               size=self.img_size, mode="nearest").squeeze(0).squeeze(0).long()
        return img_t, mask_t


train_inline_ds = F3Dataset25D(train_norm, train_labels, train_inline_idx, axis=0,
                                transform=inline_transform, augment_polarity=True)
train_xline_ds  = F3Dataset25D(train_norm, train_labels, train_xline_idx, axis=1,
                                transform=xline_transform, augment_polarity=True,
                                train_inline_mask=train_inline_mask)
train_ds = ConcatDataset([train_inline_ds, train_xline_ds])
val_ds   = F3Dataset25D(train_norm, train_labels, val_inline_idx, axis=0)
test1_ds = F3Dataset25D(test1_norm, test1_labels, np.arange(test1_norm.shape[0]), axis=0)
test2_ds = F3Dataset25D(test2_norm, test2_labels, np.arange(test2_norm.shape[1]), axis=1)


# WeightedRandomSampler — rare class boost (v9 style)
RARE_CLASSES = [4, 5]; RARE_BOOST = 15.0  # v2: 10 -> 15
def slice_weights(labels, indices, axis, mask=None):
    w = []
    for i in indices:
        sl = labels[i] if axis == 0 else (labels[:, i][mask] if mask is not None else labels[:, i])
        rare = sum(int((sl == c).sum()) for c in RARE_CLASSES)
        w.append(1.0 + (rare / sl.size) * RARE_BOOST)
    return w

w_inline = slice_weights(train_labels, train_inline_idx, 0)
w_xline  = slice_weights(train_labels, train_xline_idx, 1, mask=train_inline_mask)
sampler = WeightedRandomSampler(w_inline + w_xline, len(train_ds), replacement=True)

NUM_WORKERS = 0 if sys.platform == "win32" else 2
pin = device.type == "cuda"
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=pin)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test1_loader = DataLoader(test1_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test2_loader = DataLoader(test2_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# Class weights & focal alpha
train_counts = np.bincount(train_labels[train_inline_idx].flatten(), minlength=NUM_CLASSES).astype(float)
class_freq = train_counts / train_counts.sum()
focal_alpha = 1.0 / (class_freq + 1e-8); focal_alpha = focal_alpha / focal_alpha.sum()
focal_alpha_t = torch.FloatTensor(focal_alpha).to(device)


# ─── Losses (v9 ile ayni, QuadrupleLoss + FP32 Lovasz) ──────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', alpha); self.gamma = gamma
    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha[target] * ((1 - pt) ** self.gamma) * ce).mean()


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        p = torch.softmax(pred, 1)
        t = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dice = 0.0
        for c in range(NUM_CLASSES):
            inter = (p[:, c] * t[:, c]).sum()
            dice += (2 * inter + 1.0) / (p[:, c].sum() + t[:, c].sum() + 1.0)
        return 1 - dice / NUM_CLASSES


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    j = 1. - inter / union
    if len(gt_sorted) > 1: j[1:] = j[1:] - j[:-1]
    return j


def _lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0: return probas * 0.
    C = probas.size(1); losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0: continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else probas.sum() * 0.


class LovaszSoftmax(nn.Module):
    def forward(self, logits, labels):
        with torch.amp.autocast(device_type='cuda', enabled=False):
            logits = logits.float()
            probas = torch.softmax(logits, dim=1)
            B, C, H, W = probas.shape
            return _lovasz_softmax_flat(probas.permute(0, 2, 3, 1).reshape(-1, C), labels.reshape(-1))


class QuadrupleLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma); self.dice = DiceLoss(); self.lovasz = LovaszSoftmax()
        self.label_smoothing = label_smoothing
    def forward(self, pred, target):
        with torch.amp.autocast(device_type='cuda', enabled=False):
            pred = pred.float()
            ls_ce = F.cross_entropy(pred, target, label_smoothing=self.label_smoothing)
            return (0.30 * ls_ce + 0.25 * self.dice(pred, target)
                    + 0.25 * self.focal(pred, target) + 0.20 * self.lovasz(pred, target))


criterion = QuadrupleLoss(focal_alpha_t, gamma=2.0, label_smoothing=LABEL_SMOOTH).to(device)


# ─── Mixup ──────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ─── Model ──────────────────────────────────────────────────────────────────
print(f"\nModel kuruluyor (SFM Base-512, full fine-tune)...")
model = build_sfm_segmenter(num_classes=NUM_CLASSES, pretrained_path=PRETRAINED,
                            input_size=IMG_SIZE[0], in_channels=N_CHANNELS,
                            freeze_encoder=False).to(device)
n_total = sum(p.numel() for p in model.parameters())
print(f"  Total params: {n_total/1e6:.1f}M (full unfrozen)")


# ─── Layer-wise LR decay (MAE fine-tune standardi) ──────────────────────────
def get_lr_groups(model, base_lr_enc, lr_head, layer_decay, wd):
    """ViT blocks'a layer-wise LR decay uygula. Decoder + adapter ayri lr_head."""
    groups = {}
    # Encoder layers: patch_embed (en derin), blocks[0..11], norm (en yuzeysel)
    n_blocks = len(model.vit.blocks)
    # blocks[i] -> layer_id = i+1 (patch_embed = 0)
    # En son layer'larin lr'i daha yuksek, baslangictakiler daha dusuk
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.startswith('channel_adapter') or name.startswith('vit.decoder') or name.startswith('vit.cls'):
            scale = 1.0; lr = lr_head
        elif name.startswith('vit.patch_embed') or name.startswith('vit.pos_embed') or name.startswith('vit.cls_token'):
            layer_id = 0
            scale = layer_decay ** (n_blocks + 1 - layer_id)
            lr = base_lr_enc * scale
        elif name.startswith('vit.blocks'):
            block_idx = int(name.split('.')[2])
            layer_id = block_idx + 1
            scale = layer_decay ** (n_blocks + 1 - layer_id)
            lr = base_lr_enc * scale
        elif name.startswith('vit.norm') or name.startswith('vit.fc_norm'):
            scale = 1.0; lr = base_lr_enc  # norm en yuzeysel, full lr
        else:
            scale = 1.0; lr = base_lr_enc
        # No weight decay for bias and norm
        if 'bias' in name or 'norm' in name.lower() or name.endswith('.gamma') or name.endswith('.beta'):
            wd_g = 0.0
        else:
            wd_g = wd
        key = (lr, wd_g)
        groups.setdefault(key, []).append(param)
    param_groups = [{'params': params, 'lr': lr, 'weight_decay': wd_g} for (lr, wd_g), params in groups.items()]
    return param_groups


param_groups = get_lr_groups(model, BASE_LR_ENC, LR_HEAD, LAYER_DECAY, WEIGHT_DECAY)
print(f"  {len(param_groups)} param group (layer-wise LR decay {LAYER_DECAY})")

optimizer = torch.optim.AdamW(param_groups)
# Warmup + cosine
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda") if use_amp else None


# ─── Metrik ─────────────────────────────────────────────────────────────────
def compute_metrics(preds, targets, n=NUM_CLASSES):
    cm = confusion_matrix(targets, preds, labels=list(range(n)))
    rs = cm.sum(axis=1).astype(float)
    cacc = np.where(rs > 0, cm.diagonal() / rs, 0.0)
    ious = [float(cm[c, c] / (cm[c, :].sum() + cm[:, c].sum() - cm[c, c] + 1e-8)) for c in range(n)]
    dices = []
    for c in range(n):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        dices.append(float(2 * tp / (2 * tp + fp + fn + 1e-8)))
    return {"PA": float((preds == targets).sum() / len(targets)),
            "MCA": float(cacc.mean()), "per_class_acc": cacc.tolist(),
            "mIoU": float(np.mean(ious)), "per_class_iou": ious,
            "mean_dice": float(np.mean(dices)), "per_class_dice": dices,
            "confusion_matrix": cm.tolist()}


# ─── Resume ─────────────────────────────────────────────────────────────────
start_epoch = 0; best_val_miou = 0.0; patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_dice': [], 'lr_enc': [], 'lr_head': []}
if CHECKPOINT_PATH.exists():
    print(f"\nCheckpoint bulundu, devam ediliyor: {CHECKPOINT_PATH}")
    ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state']); optimizer.load_state_dict(ck['optimizer_state'])
    scheduler.load_state_dict(ck['scheduler_state'])
    start_epoch = ck['epoch'] + 1; best_val_miou = ck['best_val_miou']
    history = ck['history']; patience_counter = ck.get('patience_counter', 0)
    if use_amp and 'scaler_state' in ck: scaler.load_state_dict(ck['scaler_state'])
    print(f"  Ep {start_epoch}/{NUM_EPOCHS} | en iyi val mIoU: {best_val_miou:.4f}")
else:
    print(f"\nSifirdan basliyor.")
print(f"\nBatch {BATCH_SIZE} x Accum {ACCUM_STEPS} = eff {BATCH_SIZE * ACCUM_STEPS}, Epoch {NUM_EPOCHS}, Patience {PATIENCE}")
print("=" * 80)


# ─── Eğitim ─────────────────────────────────────────────────────────────────
for epoch in range(start_epoch, NUM_EPOCHS):
    t0 = time.time(); model.train(); optimizer.zero_grad()
    train_loss = 0.0
    for step, (imgs, masks) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
        use_mixup = random.random() < 0.5
        if use_mixup:
            imgs, ma, mb, lam = mixup_data(imgs, masks, MIXUP_ALPHA)
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(imgs)
            if use_mixup:
                loss = (lam * criterion(preds, ma) + (1 - lam) * criterion(preds, mb)) / ACCUM_STEPS
            else:
                loss = criterion(preds, masks) / ACCUM_STEPS
        if use_amp: scaler.scale(loss).backward()
        else: loss.backward()
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], 1.0)
                optimizer.step()
            optimizer.zero_grad()
        train_loss += loss.item() * ACCUM_STEPS
    train_loss /= len(train_loader)
    scheduler.step()

    # Val
    model.eval(); val_loss = 0.0; vp, vt = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(imgs)
            val_loss += criterion(preds, masks).item()
            vp.append(preds.argmax(1).cpu().numpy().flatten())
            vt.append(masks.cpu().numpy().flatten())
    val_loss /= len(val_loader)
    vm = compute_metrics(np.concatenate(vp), np.concatenate(vt))

    lrs = [g['lr'] for g in optimizer.param_groups]
    lr_enc = min(lrs); lr_head = max(lrs)
    history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
    history['val_miou'].append(vm['mIoU']); history['val_dice'].append(vm['mean_dice'])
    history['lr_enc'].append(lr_enc); history['lr_head'].append(lr_head)

    marker = ""
    if vm['mIoU'] > best_val_miou:
        best_val_miou = vm['mIoU']
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        marker = "  *** BEST ***"; patience_counter = 0
    else:
        patience_counter += 1

    ck = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
          'scheduler_state': scheduler.state_dict(), 'best_val_miou': best_val_miou,
          'history': history, 'patience_counter': patience_counter}
    if use_amp: ck['scaler_state'] = scaler.state_dict()
    torch.save(ck, CHECKPOINT_PATH)

    print(f"Ep {epoch+1:03d}/{NUM_EPOCHS} | T: {train_loss:.4f} | V: {val_loss:.4f} | "
          f"mIoU: {vm['mIoU']:.4f} | Dice: {vm['mean_dice']:.4f} | "
          f"lr_enc: {lr_enc:.2e} lr_head: {lr_head:.2e} | {time.time()-t0:.0f}s{marker}")
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping @ epoch {epoch+1}")
        break

print(f"\nEgitim tamamlandi. En iyi val mIoU: {best_val_miou:.4f}")


# ─── TTA eval (sade: HFlip + polarity; SFM ViT fixed img_size icin multi-scale yok) ───
print("\nEval (sade TTA: HFlip + polarity)...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
model.eval()

def run_eval(loader, tta=True):
    p, t = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            x = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
            probs = torch.softmax(logits, 1); n_aug = 1
            if tta:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    lh = model(torch.flip(x, dims=[3]))
                probs += torch.softmax(torch.flip(lh, dims=[3]), 1); n_aug += 1
                with torch.amp.autocast("cuda", enabled=use_amp):
                    ln = model(-x)
                probs += torch.softmax(ln, 1); n_aug += 1
                probs /= n_aug
            p.append(probs.argmax(1).cpu().numpy().flatten())
            t.append(masks.numpy().flatten())
    return np.concatenate(p), np.concatenate(t)


p1n, t1n = run_eval(test1_loader, tta=False)
p2n, t2n = run_eval(test2_loader, tta=False)
m1n = compute_metrics(p1n, t1n); m2n = compute_metrics(p2n, t2n)
print(f"  No-TTA: Test1 mIoU={m1n['mIoU']*100:.2f}%  Test2 mIoU={m2n['mIoU']*100:.2f}%")

p1, t1 = run_eval(test1_loader, tta=True)
p2, t2 = run_eval(test2_loader, tta=True)
m1 = compute_metrics(p1, t1); m2 = compute_metrics(p2, t2)
m_all = compute_metrics(np.concatenate([p1, p2]), np.concatenate([t1, t2]))

print("\n" + "=" * 70); print(" SFM FULL FINE-TUNE SONUC (multi-scale TTA)"); print("=" * 70)
print(f"  Test1 mIoU    : {m1['mIoU']*100:.2f}%")
print(f"  Test2 mIoU    : {m2['mIoU']*100:.2f}%")
print(f"  Combined mIoU : {m_all['mIoU']*100:.2f}%  (v9 baseline: 77.69%)")
print(f"  Combined Dice : {m_all['mean_dice']*100:.2f}%")
print(f"  Combined PA   : {m_all['PA']*100:.2f}%")
print(f"  Class 4 Test2 : {m2['per_class_iou'][4]*100:.2f}%  (v9: 23.04%)")

CLS = ["Upper NS","Lower NS","Rijnland","Scruff","Zechstein","Under Zech"]
print("\n  Per-class IoU (Combined):")
for i, n in enumerate(CLS):
    print(f"    S{i} {n:15s}: IoU={m_all['per_class_iou'][i]:.4f}")

print(f"\n  v9 ile delta:")
v9_comb=0.7769; v9_t1=0.7752; v9_t2=0.6869; v9_c4=0.2304
print(f"    Combined: {(m_all['mIoU']-v9_comb)*100:+.2f}p")
print(f"    Test1:    {(m1['mIoU']-v9_t1)*100:+.2f}p")
print(f"    Test2:    {(m2['mIoU']-v9_t2)*100:+.2f}p")
print(f"    C4 Test2: {(m2['per_class_iou'][4]-v9_c4)*100:+.2f}p")

results = {
    "model": "SFM ViT-Base/16 full fine-tune v2 (enc_lr 2e-4, layer_decay 0.9, rare_boost 15, 80 epoch)",
    "pretrained": str(PRETRAINED),
    "config": {"in_channels": N_CHANNELS, "img_size": list(IMG_SIZE),
               "batch_size": BATCH_SIZE, "accum": ACCUM_STEPS, "epochs": NUM_EPOCHS,
               "base_lr_enc": BASE_LR_ENC, "lr_head": LR_HEAD, "layer_decay": LAYER_DECAY,
               "weight_decay": WEIGHT_DECAY, "warmup": WARMUP_EPOCHS,
               "loss_weights": {"ls_ce": 0.30, "dice": 0.25, "focal": 0.25, "lovasz": 0.20}},
    "test1": m1, "test2": m2, "combined": m_all,
    "test1_no_tta": m1n, "test2_no_tta": m2n,
    "history": history,
}
out = METRICS_DIR / "sfm_finetune_v2_metrics.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrikler: {out}")
