"""v9 mimarisini farkli seed ile egit. Multi-seed ensemble icin SEED=43, 44 ile cagrilir.

v9 ana modeli (deeplabv3plus_v9.ipynb) zaten SEED=42 ile egitildi.
Bu script ayni mimari + ayni hyperparams + ayni split, sadece SEED degisir.
Cikti: checkpoints_v7/v9_seed_{SEED}_best.pth + results/metrics/v9_seed_{SEED}_metrics.json

Kullanim:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe ensemble/train_v9_seed.py 43
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe ensemble/train_v9_seed.py 44
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
import segmentation_models_pytorch as smp


# ─── CLI: SEED ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Kullanim: train_v9_seed.py <SEED>  (orn: 43)")
    sys.exit(1)
SEED = int(sys.argv[1])

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
CKPT_DIR    = PROJECT_DIR / "checkpoints_v7"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CKPT_DIR / f"v9_seed_{SEED}_checkpoint.pth"
BEST_MODEL_PATH = CKPT_DIR / f"v9_seed_{SEED}_best.pth"

# v9 hyperparams (deeplabv3plus_v9.ipynb ile birebir aynı)
IMG_SIZE     = (384, 384)
N_NEIGHBORS  = 2
N_CHANNELS   = 2 * N_NEIGHBORS + 1
NUM_CLASSES  = 6
BATCH_SIZE   = 4
ACCUM_STEPS  = 6
NUM_EPOCHS   = 100
PATIENCE     = 25
MIXUP_ALPHA  = 0.2
LABEL_SMOOTH = 0.1

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Cihaz: {device}  AMP: {use_amp}  SEED: {SEED}")


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


# ─── Augmentation (v9 ile birebir) ──────────────────────────────────────────
inline_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, border_mode=0, p=0.5),
    A.ElasticTransform(alpha=80, sigma=10, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.GaussNoise(std_range=(0.001, 0.015), p=0.3),
    A.CoarseDropout(max_holes=6, max_height=20, max_width=20, fill=0, p=0.2),
])
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
        img = np.stack(slices, axis=-1)
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


# Rare-class boost (v9 ile aynı)
RARE_CLASSES = [4, 5]; RARE_BOOST = 10.0
def slice_weights(labels, indices, axis, mask=None):
    w = []
    for i in indices:
        sl = labels[i] if axis == 0 else (labels[:, i][mask] if mask is not None else labels[:, i])
        rare = sum(int((sl == c).sum()) for c in RARE_CLASSES)
        w.append(1.0 + (rare / sl.size) * RARE_BOOST)
    return w

w_inline = slice_weights(train_labels, train_inline_idx, 0)
w_xline  = slice_weights(train_labels, train_xline_idx, 1, mask=train_inline_mask)

# torch generator with seed for sampler
g = torch.Generator(); g.manual_seed(SEED)
sampler = WeightedRandomSampler(w_inline + w_xline, len(train_ds), replacement=True, generator=g)

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


# ─── Losses (v9 QuadrupleLoss FP32 wrap) ────────────────────────────────────
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


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ─── Model (v9 ile birebir) ─────────────────────────────────────────────────
print(f"\nModel kuruluyor (v9: DeepLabV3+ + EffNet-B4 + 5ch + 384x384)...")
model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=N_CHANNELS,
    classes=NUM_CLASSES,
    activation=None,
    encoder_output_stride=16,
    decoder_atrous_rates=(12, 24, 36),
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Total: {n_params/1e6:.1f}M params")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=25, T_mult=1, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda") if use_amp else None


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
history = {'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_dice': []}
if CHECKPOINT_PATH.exists():
    print(f"\nCheckpoint resume: {CHECKPOINT_PATH}")
    ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state']); optimizer.load_state_dict(ck['optimizer_state'])
    scheduler.load_state_dict(ck['scheduler_state'])
    start_epoch = ck['epoch'] + 1; best_val_miou = ck['best_val_miou']
    history = ck['history']; patience_counter = ck.get('patience_counter', 0)
    if use_amp and 'scaler_state' in ck: scaler.load_state_dict(ck['scaler_state'])
    print(f"  Ep {start_epoch}/{NUM_EPOCHS} | best val mIoU: {best_val_miou:.4f}")
else:
    print(f"\nSifirdan basliyor (SEED={SEED}).")

print(f"Batch {BATCH_SIZE} x Accum {ACCUM_STEPS} = eff {BATCH_SIZE*ACCUM_STEPS}, Epoch {NUM_EPOCHS}, Patience {PATIENCE}")
print("=" * 80)


# ─── Egitim ─────────────────────────────────────────────────────────────────
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
        train_loss += loss.item() * ACCUM_STEPS
    train_loss /= len(train_loader)
    scheduler.step()

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
    history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
    history['val_miou'].append(vm['mIoU']); history['val_dice'].append(vm['mean_dice'])

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
          f"mIoU: {vm['mIoU']:.4f} | Dice: {vm['mean_dice']:.4f} | {time.time()-t0:.0f}s{marker}")
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping @ epoch {epoch+1}")
        break

print(f"\nSEED {SEED} egğitim tamamlandi. En iyi val mIoU: {best_val_miou:.4f}")


# ─── Multi-scale TTA Eval ───────────────────────────────────────────────────
print("\nEval (multi-scale TTA: HFlip + polarity + scale 0.75/1.0/1.25)...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
model.eval()
TTA_SCALES = [0.75, 1.25]

def run_eval(loader, tta=True):
    p, t = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            x = imgs.to(device, non_blocking=True)
            H, W = x.shape[2], x.shape[3]
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
                for sc in TTA_SCALES:
                    hs = max(32, (int(H * sc) // 16) * 16)
                    ws = max(32, (int(W * sc) // 16) * 16)
                    xs = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        ls_ = model(xs)
                    ls_up = F.interpolate(ls_, size=(H, W), mode="bilinear", align_corners=False)
                    probs += torch.softmax(ls_up, 1); n_aug += 1
                probs /= n_aug
            p.append(probs.argmax(1).cpu().numpy().flatten())
            t.append(masks.numpy().flatten())
    return np.concatenate(p), np.concatenate(t)


p1n, t1n = run_eval(test1_loader, tta=False)
p2n, t2n = run_eval(test2_loader, tta=False)
m1n = compute_metrics(p1n, t1n); m2n = compute_metrics(p2n, t2n)

p1, t1 = run_eval(test1_loader, tta=True)
p2, t2 = run_eval(test2_loader, tta=True)
m1 = compute_metrics(p1, t1); m2 = compute_metrics(p2, t2)
m_all = compute_metrics(np.concatenate([p1, p2]), np.concatenate([t1, t2]))

print(f"\nSEED {SEED} sonuc: Combined mIoU = {m_all['mIoU']*100:.2f}% | "
      f"Test1 {m1['mIoU']*100:.2f}% | Test2 {m2['mIoU']*100:.2f}% | "
      f"C4 Test2 {m2['per_class_iou'][4]*100:.2f}%")

results = {
    "model": f"v9 mimari (DeepLabV3+ EffNet-B4 5ch 384x384) SEED={SEED}",
    "seed": SEED,
    "config": {"in_channels": N_CHANNELS, "img_size": list(IMG_SIZE),
               "batch_size": BATCH_SIZE, "accum": ACCUM_STEPS, "epochs": NUM_EPOCHS,
               "loss_weights": {"ls_ce": 0.30, "dice": 0.25, "focal": 0.25, "lovasz": 0.20}},
    "test1": m1, "test2": m2, "combined": m_all,
    "test1_no_tta": m1n, "test2_no_tta": m2n,
    "history": history,
}
out = METRICS_DIR / f"v9_seed_{SEED}_metrics.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrikler: {out}")
