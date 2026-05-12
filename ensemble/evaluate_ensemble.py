"""Multi-seed ensemble inference: 3 best model (SEED 42, 43, 44) softmax averaging.

v9 SEED=42 mevcut: checkpoints_v7/deeplabv3plus_v9_best.pth
v9 SEED=43 mevcut: checkpoints_v7/v9_seed_43_best.pth
v9 SEED=44 mevcut: checkpoints_v7/v9_seed_44_best.pth (egğitim biter bitmez)

Strateji:
1. Her model icin sirayla yukle, Test1+Test2'de multi-scale TTA ile softmax cikar
2. Softmax tensor'lerini RAM'de biriktir (3 model x ~350MB = ~1GB, OK)
3. Softmax average -> argmax -> metric
4. Tekil ve ensemble sonuc bir tablo
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
CKPT_DIR    = PROJECT_DIR / "checkpoints_v7"

# Model checkpoint'leri
MODELS = {
    42: CKPT_DIR / "deeplabv3plus_v9_best.pth",
    43: CKPT_DIR / "v9_seed_43_best.pth",
    44: CKPT_DIR / "v9_seed_44_best.pth",
}

IMG_SIZE    = (384, 384)
N_NEIGHBORS = 2
N_CHANNELS  = 2 * N_NEIGHBORS + 1
NUM_CLASSES = 6
BATCH_SIZE  = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Cihaz: {device}  AMP: {use_amp}")


# ─── Veri ───────────────────────────────────────────────────────────────────
print("\nVeri yukleniyor...")
train_seismic = np.load(DATA_DIR / "train" / "train_seismic.npy")
test1_seismic = np.load(DATA_DIR / "test_once" / "test1_seismic.npy")
test1_labels  = np.load(DATA_DIR / "test_once" / "test1_labels.npy")
test2_seismic = np.load(DATA_DIR / "test_once" / "test2_seismic.npy")
test2_labels  = np.load(DATA_DIR / "test_once" / "test2_labels.npy")

train_mean = train_seismic.mean(); train_std = train_seismic.std() + 1e-8
test1_norm = ((test1_seismic - train_mean) / train_std).astype(np.float32)
test2_norm = ((test2_seismic - train_mean) / train_std).astype(np.float32)
del train_seismic, test1_seismic, test2_seismic


class F3Dataset25D(Dataset):
    def __init__(self, volume, labels, axis=0):
        self.volume = volume; self.labels = labels; self.axis = axis
        self.n_slices = volume.shape[axis]

    def __len__(self): return self.n_slices

    def __getitem__(self, idx):
        slices = []
        for k in range(-N_NEIGHBORS, N_NEIGHBORS + 1):
            j = max(0, min(self.n_slices - 1, idx + k))
            s = self.volume[j] if self.axis == 0 else self.volume[:, j]
            slices.append(s.astype(np.float32))
        img = np.stack(slices, axis=-1)
        mask = self.labels[idx] if self.axis == 0 else self.labels[:, idx]
        mask = mask.astype(np.int64)
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask.copy())
        img_t = F.interpolate(img_t.unsqueeze(0), size=IMG_SIZE, mode="bilinear",
                              align_corners=False).squeeze(0)
        mask_t = F.interpolate(mask_t.float().unsqueeze(0).unsqueeze(0),
                               size=IMG_SIZE, mode="nearest").squeeze(0).squeeze(0).long()
        return img_t, mask_t


test1_loader = DataLoader(F3Dataset25D(test1_norm, test1_labels, 0), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test2_loader = DataLoader(F3Dataset25D(test2_norm, test2_labels, 1), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def build_model():
    return smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4", encoder_weights=None, in_channels=N_CHANNELS,
        classes=NUM_CLASSES, activation=None, encoder_output_stride=16,
        decoder_atrous_rates=(12, 24, 36),
    ).to(device)


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
            "mean_dice": float(np.mean(dices)), "per_class_dice": dices}


def run_model_tta(model, loader):
    """Multi-scale TTA inference; donus: softmax tensor (N, C, H, W) ve targets (N, H, W)."""
    TTA_SCALES = [0.75, 1.25]
    all_softmax = []; all_targets = []
    model.eval()
    with torch.no_grad():
        for imgs, masks in loader:
            x = imgs.to(device, non_blocking=True)
            H, W = x.shape[2], x.shape[3]
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
            probs = torch.softmax(logits, 1); n_aug = 1
            # HFlip
            with torch.amp.autocast("cuda", enabled=use_amp):
                lh = model(torch.flip(x, dims=[3]))
            probs += torch.softmax(torch.flip(lh, dims=[3]), 1); n_aug += 1
            # Polarity
            with torch.amp.autocast("cuda", enabled=use_amp):
                ln = model(-x)
            probs += torch.softmax(ln, 1); n_aug += 1
            # Multi-scale
            for sc in TTA_SCALES:
                hs = max(32, (int(H * sc) // 16) * 16)
                ws = max(32, (int(W * sc) // 16) * 16)
                xs = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    ls_ = model(xs)
                ls_up = F.interpolate(ls_, size=(H, W), mode="bilinear", align_corners=False)
                probs += torch.softmax(ls_up, 1); n_aug += 1
            probs /= n_aug
            all_softmax.append(probs.float().cpu().numpy())
            all_targets.append(masks.numpy())
    return np.concatenate(all_softmax, axis=0), np.concatenate(all_targets, axis=0)


# ─── Her model icin softmax cikti ───────────────────────────────────────────
softmaxes_t1 = {}; softmaxes_t2 = {}; targets_t1 = None; targets_t2 = None
individual_results = {}

for seed, ckpt_path in MODELS.items():
    if not ckpt_path.exists():
        print(f"\n⚠️ SEED {seed} checkpoint yok: {ckpt_path} — atlandi")
        continue
    print(f"\n[SEED {seed}] Model yukleniyor: {ckpt_path.name}")
    model = build_model()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    print(f"  Test1 inference (multi-scale TTA)...")
    sm1, tg1 = run_model_tta(model, test1_loader)
    print(f"  Test2 inference (multi-scale TTA)...")
    sm2, tg2 = run_model_tta(model, test2_loader)

    softmaxes_t1[seed] = sm1; softmaxes_t2[seed] = sm2
    if targets_t1 is None:
        targets_t1 = tg1; targets_t2 = tg2

    # Bu modelin tekil sonuclari
    pred_t1 = sm1.argmax(1).reshape(-1); flat_t1 = tg1.reshape(-1)
    pred_t2 = sm2.argmax(1).reshape(-1); flat_t2 = tg2.reshape(-1)
    m1 = compute_metrics(pred_t1, flat_t1)
    m2 = compute_metrics(pred_t2, flat_t2)
    m_all = compute_metrics(np.concatenate([pred_t1, pred_t2]),
                            np.concatenate([flat_t1, flat_t2]))
    individual_results[seed] = {"test1": m1, "test2": m2, "combined": m_all}
    print(f"  SEED {seed}: Combined mIoU={m_all['mIoU']*100:.2f}%  "
          f"Test1={m1['mIoU']*100:.2f}%  Test2={m2['mIoU']*100:.2f}%  "
          f"C4 Test2={m2['per_class_iou'][4]*100:.2f}%")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

if len(softmaxes_t1) < 2:
    print("\n⚠️ En az 2 model gerekli ensemble icin. Sadece bulunan model(ler) raporlandi.")
    sys.exit(0)


# ─── Ensemble: softmax averaging ────────────────────────────────────────────
print(f"\n[ENSEMBLE] Softmax averaging ({len(softmaxes_t1)} model)...")
ens_sm1 = np.mean(list(softmaxes_t1.values()), axis=0)
ens_sm2 = np.mean(list(softmaxes_t2.values()), axis=0)

ens_pred_t1 = ens_sm1.argmax(1).reshape(-1); flat_t1 = targets_t1.reshape(-1)
ens_pred_t2 = ens_sm2.argmax(1).reshape(-1); flat_t2 = targets_t2.reshape(-1)

ens_m1 = compute_metrics(ens_pred_t1, flat_t1)
ens_m2 = compute_metrics(ens_pred_t2, flat_t2)
ens_all = compute_metrics(np.concatenate([ens_pred_t1, ens_pred_t2]),
                          np.concatenate([flat_t1, flat_t2]))


# ─── Karsilastirma tablosu ──────────────────────────────────────────────────
print("\n" + "=" * 75)
print(" MULTI-SEED ENSEMBLE SONUC")
print("=" * 75)

v9_baseline = {"combined": 0.7769, "test1": 0.7752, "test2": 0.6869, "c4_test2": 0.2304}
print(f"\n{'Model':<20} {'Combined':>10} {'Test1':>10} {'Test2':>10} {'C4 Test2':>10}")
print("-" * 75)
print(f"{'v9 (baseline ref)':<20} {v9_baseline['combined']*100:>9.2f}%  {v9_baseline['test1']*100:>9.2f}%  "
      f"{v9_baseline['test2']*100:>9.2f}%  {v9_baseline['c4_test2']*100:>9.2f}%")
for seed, r in individual_results.items():
    print(f"{'SEED ' + str(seed):<20} {r['combined']['mIoU']*100:>9.2f}%  {r['test1']['mIoU']*100:>9.2f}%  "
          f"{r['test2']['mIoU']*100:>9.2f}%  {r['test2']['per_class_iou'][4]*100:>9.2f}%")
print("-" * 75)
print(f"{'★ ENSEMBLE':<20} {ens_all['mIoU']*100:>9.2f}%  {ens_m1['mIoU']*100:>9.2f}%  "
      f"{ens_m2['mIoU']*100:>9.2f}%  {ens_m2['per_class_iou'][4]*100:>9.2f}%")
print("=" * 75)

print(f"\nDelta (ensemble vs v9):")
print(f"  Combined: {(ens_all['mIoU']-v9_baseline['combined'])*100:+.2f}p")
print(f"  Test1:    {(ens_m1['mIoU']-v9_baseline['test1'])*100:+.2f}p")
print(f"  Test2:    {(ens_m2['mIoU']-v9_baseline['test2'])*100:+.2f}p")
print(f"  C4 Test2: {(ens_m2['per_class_iou'][4]-v9_baseline['c4_test2'])*100:+.2f}p")

# Variance / std
miou_vals = [r['combined']['mIoU'] for r in individual_results.values()]
print(f"\nSeed varyansi (Combined mIoU): mean={np.mean(miou_vals):.4f}  std={np.std(miou_vals):.4f}")

CLS = ["Upper NS","Lower NS","Rijnland","Scruff","Zechstein","Under Zech"]
print(f"\nPer-class IoU (Combined) — ENSEMBLE:")
for i, n in enumerate(CLS):
    print(f"  S{i} {n:15s}: IoU={ens_all['per_class_iou'][i]:.4f}")


# ─── Kaydet ─────────────────────────────────────────────────────────────────
results = {
    "model": f"v9 multi-seed ensemble ({len(softmaxes_t1)} models, softmax averaging, multi-scale TTA)",
    "seeds": list(softmaxes_t1.keys()),
    "individual": {str(s): r for s, r in individual_results.items()},
    "ensemble": {"test1": ens_m1, "test2": ens_m2, "combined": ens_all},
    "variance": {"miou_mean": float(np.mean(miou_vals)), "miou_std": float(np.std(miou_vals))},
    "v9_baseline_ref": v9_baseline,
}
out = METRICS_DIR / "v9_ensemble_metrics.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrikler: {out}")
