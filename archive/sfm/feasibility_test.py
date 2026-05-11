"""SFM fizibilite testi: 10 epoch hizli eğitim + eval.

Bu script:
- v9 dataset (5-kanal 2.5D, Yol A methodology-fixed split) kullanir
- SFM ViT-Base/16 encoder (frozen) + 5->3 channel adapter + MLAHead decoder
- 10 epoch hizli egitim (sadece channel_adapter + decoder trainable, ~4.2M params)
- Test1 ve Test2'de eval (TTA yok)
- v9 ile karsilastirma

Kullanim:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe sfm/feasibility_test.py

Cikti:
  - results/metrics/sfm_feasibility_metrics.json
  - Konsola progres ve karsilastirma tablosu
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import confusion_matrix

# SFM kodu
sys.path.insert(0, str(Path(__file__).parent / "code"))
from sfm_wrapper import build_sfm_segmenter

# ─── Konfig ─────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PRETRAINED  = PROJECT_DIR / "sfm" / "pretrained" / "SFM-Base-512.pth"

SEED         = 42
IMG_SIZE     = (384, 384)
N_NEIGHBORS  = 2
N_CHANNELS   = 2 * N_NEIGHBORS + 1
BATCH_SIZE   = 4
ACCUM_STEPS  = 6
NUM_EPOCHS   = 10           # hizli fizibilite
LR_HEAD      = 1e-3         # frozen encoder ile head egitilir, daha yuksek lr
NUM_CLASSES  = 6
PATIENCE     = 5

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

# Yol A methodology fix
n_inlines = train_seismic.shape[0]; n_xlines = train_seismic.shape[1]
BUFFER = 2
val_start, val_end = 160, 240
val_inline_idx   = np.arange(val_start, val_end)
train_inline_idx = np.concatenate([np.arange(0, val_start - BUFFER),
                                   np.arange(val_end + BUFFER, n_inlines)])
train_xline_idx  = np.arange(0, n_xlines)
train_inline_mask = np.zeros(n_inlines, dtype=bool)
train_inline_mask[train_inline_idx] = True

# Normalize
train_mean = train_seismic.mean(); train_std = train_seismic.std() + 1e-8
train_norm = ((train_seismic - train_mean) / train_std).astype(np.float32)
test1_norm = ((test1_seismic - train_mean) / train_std).astype(np.float32)
test2_norm = ((test2_seismic - train_mean) / train_std).astype(np.float32)
del train_seismic, test1_seismic, test2_seismic

print(f"  Train inline: {len(train_inline_idx)}, val: {len(val_inline_idx)}, xline: {len(train_xline_idx)}")
print(f"  Test1: {test1_norm.shape}, Test2: {test2_norm.shape}")


# ─── Dataset (5-kanal 2.5D, methodology-fix mask) ──────────────────────────
class F3Dataset25D(Dataset):
    def __init__(self, volume, labels, indices, axis=0, img_size=IMG_SIZE,
                 train_inline_mask=None, n_neighbors=N_NEIGHBORS):
        self.volume = volume; self.labels = labels; self.indices = indices
        self.axis = axis; self.n_slices = volume.shape[axis]
        self.img_size = img_size; self.n_neighbors = n_neighbors
        self.train_inline_mask = train_inline_mask

    def _slice(self, vol, idx):
        if self.axis == 0:
            return vol[idx]
        s = vol[:, idx]
        if self.train_inline_mask is not None:
            s = s[self.train_inline_mask]
        return s

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        slices = []
        for k in range(-self.n_neighbors, self.n_neighbors + 1):
            j = max(0, min(self.n_slices - 1, i + k))
            slices.append(self._slice(self.volume, j).astype(np.float32))
        img = np.stack(slices, axis=-1)
        mask = self._slice(self.labels, i).astype(np.int64)

        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask.copy())
        img_t = F.interpolate(img_t.unsqueeze(0), size=self.img_size,
                              mode="bilinear", align_corners=False).squeeze(0)
        mask_t = F.interpolate(mask_t.float().unsqueeze(0).unsqueeze(0),
                               size=self.img_size, mode="nearest").squeeze(0).squeeze(0).long()
        return img_t, mask_t


train_inline_ds = F3Dataset25D(train_norm, train_labels, train_inline_idx, axis=0)
train_xline_ds  = F3Dataset25D(train_norm, train_labels, train_xline_idx, axis=1,
                                train_inline_mask=train_inline_mask)
train_ds = ConcatDataset([train_inline_ds, train_xline_ds])
val_ds   = F3Dataset25D(train_norm, train_labels, val_inline_idx, axis=0)
test1_ds = F3Dataset25D(test1_norm, test1_labels, np.arange(test1_norm.shape[0]), axis=0)
test2_ds = F3Dataset25D(test2_norm, test2_labels, np.arange(test2_norm.shape[1]), axis=1)

NUM_WORKERS = 0 if sys.platform == "win32" else 2
pin = device.type == "cuda"
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=pin)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test1_loader = DataLoader(test1_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test2_loader = DataLoader(test2_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f"  Train batch: {len(train_loader)}, Val: {len(val_loader)}, "
      f"Test1: {len(test1_loader)}, Test2: {len(test2_loader)}")


# ─── Class weights (sadece train inline'dan) ────────────────────────────────
train_counts = np.bincount(train_labels[train_inline_idx].flatten(), minlength=NUM_CLASSES).astype(float)
class_freq = train_counts / train_counts.sum()
class_weight = 1.0 / (class_freq + 1e-8)
class_weight = class_weight / class_weight.sum() * NUM_CLASSES
class_weight_t = torch.FloatTensor(class_weight).to(device)
print(f"  Class weights: {[f'{w:.2f}' for w in class_weight]}")


# ─── Loss: basit CE + Dice (Lovasz/Focal yok, hizli fizibilite) ────────────
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        p = torch.softmax(pred, 1)
        t = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dice = 0.0
        for c in range(NUM_CLASSES):
            inter = (p[:, c] * t[:, c]).sum()
            dice += (2 * inter + 1.0) / (p[:, c].sum() + t[:, c].sum() + 1.0)
        return 1 - dice / NUM_CLASSES


ce_loss = nn.CrossEntropyLoss(weight=class_weight_t, label_smoothing=0.1)
dice_loss = DiceLoss()
def criterion(pred, target):
    with torch.amp.autocast(device_type='cuda', enabled=False):
        pred = pred.float()
        return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)


# ─── Model ──────────────────────────────────────────────────────────────────
print(f"\nModel kuruluyor... (pretrained: {PRETRAINED.exists()})")
model = build_sfm_segmenter(
    num_classes=NUM_CLASSES,
    pretrained_path=PRETRAINED if PRETRAINED.exists() else None,
    input_size=IMG_SIZE[0], in_channels=N_CHANNELS,
    freeze_encoder=True,
).to(device)


# ─── Metrik ─────────────────────────────────────────────────────────────────
def compute_metrics(preds, targets, n=NUM_CLASSES):
    cm = confusion_matrix(targets, preds, labels=list(range(n)))
    ious = []
    for c in range(n):
        inter = cm[c, c]; union = cm[c, :].sum() + cm[:, c].sum() - inter
        ious.append(float(inter / (union + 1e-8)))
    dices = []
    for c in range(n):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        dices.append(float(2 * tp / (2 * tp + fp + fn + 1e-8)))
    return {
        "PA": float((preds == targets).sum() / len(targets)),
        "mIoU": float(np.mean(ious)),
        "per_class_iou": ious,
        "mean_dice": float(np.mean(dices)),
        "per_class_dice": dices,
        "confusion_matrix": cm.tolist(),
    }


# ─── Egğitim ───────────────────────────────────────────────────────────────
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR_HEAD, weight_decay=1e-3)
scaler = torch.amp.GradScaler("cuda") if use_amp else None

print(f"\nEgitim basliyor: {NUM_EPOCHS} epoch, batch {BATCH_SIZE} x accum {ACCUM_STEPS}")
print("=" * 70)
best_val_miou = 0.0
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'val_miou': []}

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    train_loss = 0.0
    for step, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                preds = model(imgs)
                loss = criterion(preds, masks) / ACCUM_STEPS
            scaler.scale(loss).backward()
        else:
            preds = model(imgs)
            loss = criterion(preds, masks) / ACCUM_STEPS
            loss.backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
            optimizer.zero_grad()
        train_loss += loss.item() * ACCUM_STEPS
    train_loss /= len(train_loader)

    # Val
    model.eval()
    val_loss = 0.0; vp, vt = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(imgs)
            else:
                preds = model(imgs)
            val_loss += criterion(preds, masks).item()
            vp.append(preds.argmax(1).cpu().numpy().flatten())
            vt.append(masks.cpu().numpy().flatten())
    val_loss /= len(val_loader)
    vm = compute_metrics(np.concatenate(vp), np.concatenate(vt))
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_miou'].append(vm['mIoU'])

    marker = ""
    if vm['mIoU'] > best_val_miou:
        best_val_miou = vm['mIoU']
        torch.save(model.state_dict(), PROJECT_DIR / "checkpoints_v7" / "sfm_feasibility_best.pth")
        marker = "  *** BEST ***"
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Ep {epoch+1:02d}/{NUM_EPOCHS} | T: {train_loss:.4f} | V: {val_loss:.4f} | "
          f"mIoU: {vm['mIoU']:.4f} | {time.time()-t0:.0f}s{marker}")
    if patience_counter >= PATIENCE:
        print(f"Early stopping @ epoch {epoch+1}")
        break

print("\nEgitim tamamlandi. En iyi val mIoU:", best_val_miou)


# ─── Eval Test1 + Test2 ─────────────────────────────────────────────────────
print("\nTest evaluator...")
best_ckpt = PROJECT_DIR / "checkpoints_v7" / "sfm_feasibility_best.pth"
model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
model.eval()

def run_eval(loader):
    p, t = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
            else:
                logits = model(imgs)
            p.append(logits.argmax(1).cpu().numpy().flatten())
            t.append(masks.numpy().flatten())
    return np.concatenate(p), np.concatenate(t)

p1, t1 = run_eval(test1_loader)
p2, t2 = run_eval(test2_loader)
m1 = compute_metrics(p1, t1)
m2 = compute_metrics(p2, t2)
m_all = compute_metrics(np.concatenate([p1, p2]), np.concatenate([t1, t2]))

print("\n" + "=" * 70)
print(" SFM FIZIBILITE SONUC")
print("=" * 70)
print(f"  Test1 mIoU    : {m1['mIoU']*100:.2f}%")
print(f"  Test2 mIoU    : {m2['mIoU']*100:.2f}%")
print(f"  Combined mIoU : {m_all['mIoU']*100:.2f}%")
print(f"  Combined Dice : {m_all['mean_dice']*100:.2f}%")
print(f"  Combined PA   : {m_all['PA']*100:.2f}%")

CLS = ["Upper NS","Lower NS","Rijnland","Scruff","Zechstein","Under Zech"]
print("\n  Per-class IoU (Combined):")
for i, n in enumerate(CLS):
    print(f"    S{i} {n:15s}: IoU={m_all['per_class_iou'][i]:.4f}")

print("\n  v9 ile karsilastirma (baseline):")
v9_combined = 0.7769; v9_test1 = 0.7752; v9_test2 = 0.6869; v9_c4_test2 = 0.2304
print(f"    Combined: SFM {m_all['mIoU']:.4f} vs v9 {v9_combined:.4f}  "
      f"(delta {(m_all['mIoU']-v9_combined)*100:+.2f}p)")
print(f"    Test2:    SFM {m2['mIoU']:.4f} vs v9 {v9_test2:.4f}  "
      f"(delta {(m2['mIoU']-v9_test2)*100:+.2f}p)")
print(f"    C4 Test2: SFM {m2['per_class_iou'][4]:.4f} vs v9 {v9_c4_test2:.4f}  "
      f"(delta {(m2['per_class_iou'][4]-v9_c4_test2)*100:+.2f}p)")

results = {
    "model": f"SFM ViT-Base/16 (frozen encoder, {NUM_EPOCHS} epoch fizibilite)",
    "pretrained_loaded": PRETRAINED.exists(),
    "config": {
        "in_channels": N_CHANNELS, "img_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS, "epochs": NUM_EPOCHS,
        "lr": LR_HEAD, "frozen_encoder": True,
    },
    "test1": m1, "test2": m2, "combined": m_all,
    "history": history,
}
out_path = METRICS_DIR / "sfm_feasibility_metrics.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrikler: {out_path}")
