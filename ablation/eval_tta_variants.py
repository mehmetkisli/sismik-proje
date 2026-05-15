"""TTA varyant ablation — mevcut v9 best checkpoint'i kullanarak 3 TTA konfigi karşılaştır.

Eğitim gerektirmez. Sadece inference. ~30 dk RTX 3060 Ti.

Varyantlar:
  1. no-TTA      → ham softmax
  2. hflip-only  → 2 augmentation (orig + HFlip)
  3. multi-scale → 5 augmentation (orig + HFlip + polarity + scale 0.75 + scale 1.25)  ← v9 default

Çıktı:
  results/metrics/ablation_tta_variants.json

Kullanım:
  PYTHONIOENCODING=utf-8 venv/bin/python ablation/eval_tta_variants.py
  # belirli checkpoint kullan:
  PYTHONIOENCODING=utf-8 venv/bin/python ablation/eval_tta_variants.py --ckpt checkpoints_v7/deeplabv3plus_v9_best.pth
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str,
                    default="checkpoints_v7/deeplabv3plus_v9_best.pth",
                    help="Best model checkpoint (default v9 ana)")
args = parser.parse_args()

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = PROJECT_DIR / args.ckpt
assert CKPT_PATH.exists(), f"Checkpoint bulunamadi: {CKPT_PATH}"

IMG_SIZE = (384, 384)
N_NEIGHBORS = 2
N_CHANNELS = 2 * N_NEIGHBORS + 1
NUM_CLASSES = 6
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Cihaz: {device}  AMP: {use_amp}")
print(f"Checkpoint: {CKPT_PATH}")


# ─── Veri ───────────────────────────────────────────────────────────────────
train_seismic = np.load(DATA_DIR / "train" / "train_seismic.npy")
test1_seismic = np.load(DATA_DIR / "test_once" / "test1_seismic.npy")
test1_labels = np.load(DATA_DIR / "test_once" / "test1_labels.npy")
test2_seismic = np.load(DATA_DIR / "test_once" / "test2_seismic.npy")
test2_labels = np.load(DATA_DIR / "test_once" / "test2_labels.npy")

train_mean = train_seismic.mean(); train_std = train_seismic.std() + 1e-8
test1_norm = ((test1_seismic - train_mean) / train_std).astype(np.float32)
test2_norm = ((test2_seismic - train_mean) / train_std).astype(np.float32)


class F3Dataset25D(Dataset):
    def __init__(self, volume, labels, indices, axis=0, img_size=IMG_SIZE, n_neighbors=N_NEIGHBORS):
        self.volume = volume; self.labels = labels; self.indices = indices
        self.axis = axis; self.n_slices = volume.shape[axis]
        self.img_size = img_size; self.n_neighbors = n_neighbors

    def _slice(self, vol, idx):
        return vol[idx] if self.axis == 0 else vol[:, idx]

    def __len__(self): return len(self.indices)

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
        img_t = F.interpolate(img_t.unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=False).squeeze(0)
        mask_t = F.interpolate(mask_t.float().unsqueeze(0).unsqueeze(0),
                               size=self.img_size, mode="nearest").squeeze(0).squeeze(0).long()
        return img_t, mask_t


test1_ds = F3Dataset25D(test1_norm, test1_labels, np.arange(test1_norm.shape[0]), axis=0)
test2_ds = F3Dataset25D(test2_norm, test2_labels, np.arange(test2_norm.shape[1]), axis=1)
test1_loader = DataLoader(test1_ds, batch_size=BATCH_SIZE, shuffle=False)
test2_loader = DataLoader(test2_ds, batch_size=BATCH_SIZE, shuffle=False)


# ─── Model ──────────────────────────────────────────────────────────────────
print("\nModel yukleniyor...")
model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=N_CHANNELS,
    classes=NUM_CLASSES,
    activation=None,
    encoder_output_stride=16,
    decoder_atrous_rates=(12, 24, 36),
).to(device)
state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()


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
            "MCA": float(cacc.mean()),
            "mIoU": float(np.mean(ious)), "per_class_iou": ious,
            "mean_dice": float(np.mean(dices))}


def run_eval(loader, mode):
    """mode: 'none' | 'hflip' | 'multiscale'"""
    p, t = [], []
    TTA_SCALES = [0.75, 1.25]
    with torch.no_grad():
        for imgs, masks in loader:
            x = imgs.to(device, non_blocking=True)
            H, W = x.shape[2], x.shape[3]
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
            probs = torch.softmax(logits, 1); n_aug = 1

            if mode in ("hflip", "multiscale"):
                with torch.amp.autocast("cuda", enabled=use_amp):
                    lh = model(torch.flip(x, dims=[3]))
                probs += torch.softmax(torch.flip(lh, dims=[3]), 1); n_aug += 1

            if mode == "multiscale":
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


modes = [
    ("none", "Hiç TTA yok (ham softmax)"),
    ("hflip", "HFlip only (2 aug)"),
    ("multiscale", "Multi-scale full (5 aug: HFlip+polarity+scale 0.75/1.25) — v9 default"),
]

all_results = {}
for mode, desc in modes:
    print(f"\n→ {desc}")
    p1, t1 = run_eval(test1_loader, mode)
    p2, t2 = run_eval(test2_loader, mode)
    m1 = compute_metrics(p1, t1); m2 = compute_metrics(p2, t2)
    m_all = compute_metrics(np.concatenate([p1, p2]), np.concatenate([t1, t2]))
    all_results[mode] = {"description": desc, "test1": m1, "test2": m2, "combined": m_all}
    print(f"  Combined mIoU = {m_all['mIoU']*100:.2f}% | "
          f"Test1 {m1['mIoU']*100:.2f}% | Test2 {m2['mIoU']*100:.2f}% | "
          f"C4 Test2 {m2['per_class_iou'][4]*100:.2f}%")

# Karşılaştırma özeti
base_mode = "multiscale"
base_miou = all_results[base_mode]["combined"]["mIoU"]
summary = {}
for mode, _ in modes:
    miou = all_results[mode]["combined"]["mIoU"]
    summary[mode] = {
        "combined_mIoU": miou,
        "delta_vs_multiscale": miou - base_miou,
    }

print("\n" + "=" * 60)
print("TTA Ablation Özeti (Combined mIoU)")
print("=" * 60)
for mode, _ in modes:
    s = summary[mode]
    flag = " ← v9 default" if mode == base_mode else ""
    print(f"  {mode:12s}: {s['combined_mIoU']*100:.2f}%  "
          f"(Δ {s['delta_vs_multiscale']*100:+.2f}p){flag}")

out = METRICS_DIR / "ablation_tta_variants.json"
with open(out, "w") as f:
    json.dump({"checkpoint": str(CKPT_PATH), "results": all_results, "summary": summary},
              f, indent=2)
print(f"\nMetrikler: {out}")
