"""SFM eval-only: best checkpoint'i yukle, sade TTA (HFlip + polarity) ile final degerlendir.

Multi-scale TTA, ViT'in fixed img_size kontrol'u nedeniyle calismadi
(0.75x = 288 boyut patch_embed strict check fail). Bunun yerine sade TTA:
  - Scale 1.0 (orijinal)
  - HFlip
  - Polarity inversion
v9'da multi-scale TTA katkisi zaten Test2'de +0.14p idi, kayip kabul edilebilir.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent / "code"))
from sfm_wrapper import build_sfm_segmenter

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
BEST_MODEL  = PROJECT_DIR / "checkpoints_v7" / "sfm_finetune_best.pth"
PRETRAINED  = PROJECT_DIR / "sfm" / "pretrained" / "SFM-Base-512.pth"

IMG_SIZE    = (384, 384)
N_NEIGHBORS = 2
N_CHANNELS  = 2 * N_NEIGHBORS + 1
NUM_CLASSES = 6
BATCH_SIZE  = 2

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

    def __len__(self):
        return self.n_slices

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


test1_loader = DataLoader(F3Dataset25D(test1_norm, test1_labels, axis=0),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test2_loader = DataLoader(F3Dataset25D(test2_norm, test2_labels, axis=1),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ─── Model + best checkpoint ────────────────────────────────────────────────
print(f"\nModel kuruluyor...")
model = build_sfm_segmenter(num_classes=NUM_CLASSES, pretrained_path=PRETRAINED,
                            input_size=IMG_SIZE[0], in_channels=N_CHANNELS,
                            freeze_encoder=False).to(device)
print(f"\nBest checkpoint yukleniyor: {BEST_MODEL}")
model.load_state_dict(torch.load(BEST_MODEL, map_location=device, weights_only=True))
model.eval()


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


# ─── Eval (sade TTA: scale 1.0 + HFlip + polarity) ─────────────────────────
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


print("\nEval Test1 + Test2 (no TTA)...")
p1n, t1n = run_eval(test1_loader, tta=False)
p2n, t2n = run_eval(test2_loader, tta=False)
m1n = compute_metrics(p1n, t1n); m2n = compute_metrics(p2n, t2n)
print(f"  No-TTA: Test1 mIoU={m1n['mIoU']*100:.2f}%  Test2 mIoU={m2n['mIoU']*100:.2f}%")

print("\nEval Test1 + Test2 (sade TTA)...")
p1, t1 = run_eval(test1_loader, tta=True)
p2, t2 = run_eval(test2_loader, tta=True)
m1 = compute_metrics(p1, t1); m2 = compute_metrics(p2, t2)
m_all = compute_metrics(np.concatenate([p1, p2]), np.concatenate([t1, t2]))

print("\n" + "=" * 70); print(" SFM FULL FINE-TUNE FINAL (sade TTA)"); print("=" * 70)
print(f"  Test1 mIoU    : {m1['mIoU']*100:.2f}%  (v9: 77.52%)")
print(f"  Test2 mIoU    : {m2['mIoU']*100:.2f}%  (v9: 68.69%)")
print(f"  Combined mIoU : {m_all['mIoU']*100:.2f}%  (v9: 77.69%)")
print(f"  Combined Dice : {m_all['mean_dice']*100:.2f}%  (v9: 86.75%)")
print(f"  Combined PA   : {m_all['PA']*100:.2f}%  (v9: 93.45%)")
print(f"  Combined MCA  : {m_all['MCA']*100:.2f}%  (v9: 86.27%)")

CLS = ["Upper NS","Lower NS","Rijnland","Scruff","Zechstein","Under Zech"]
v9_pcl = [0.9433, 0.7968, 0.9356, 0.6008, 0.7838, 0.6010]
v9_pcl_t2 = [0.9466, 0.8083, 0.9376, 0.5464, 0.2304, 0.6510]
print("\n  Per-class IoU (Combined) — SFM vs v9:")
for i, n in enumerate(CLS):
    sfm = m_all['per_class_iou'][i]; v9 = v9_pcl[i]
    print(f"    S{i} {n:15s}: SFM {sfm:.4f}  v9 {v9:.4f}  ({(sfm-v9)*100:+.2f}p)")

print(f"\n  Per-class IoU (Test2 — kritik) — SFM vs v9:")
for i, n in enumerate(CLS):
    sfm = m2['per_class_iou'][i]; v9 = v9_pcl_t2[i]
    print(f"    S{i} {n:15s}: SFM {sfm:.4f}  v9 {v9:.4f}  ({(sfm-v9)*100:+.2f}p)")

v9_comb=0.7769; v9_t1=0.7752; v9_t2=0.6869; v9_c4_t2=0.2304
print(f"\n  v9 ile delta (sunum highlight):")
print(f"    Combined mIoU:   {(m_all['mIoU']-v9_comb)*100:+.2f}p")
print(f"    Test1 mIoU:      {(m1['mIoU']-v9_t1)*100:+.2f}p")
print(f"    Test2 mIoU:      {(m2['mIoU']-v9_t2)*100:+.2f}p")
print(f"    Class 4 Test2:   {(m2['per_class_iou'][4]-v9_c4_t2)*100:+.2f}p")

# Mevcut metrics dosyasini guncelle veya yenisini olustur
out = METRICS_DIR / "sfm_finetune_metrics.json"
if out.exists():
    existing = json.loads(out.read_text(encoding='utf-8'))
else:
    existing = {"model": "SFM ViT-Base/16 full fine-tune (simple TTA)",
                "config": {"in_channels": N_CHANNELS, "img_size": list(IMG_SIZE), "tta": "HFlip + polarity"}}

existing["test1"] = m1; existing["test2"] = m2; existing["combined"] = m_all
existing["test1_no_tta"] = m1n; existing["test2_no_tta"] = m2n
out.write_text(json.dumps(existing, indent=2), encoding='utf-8')
print(f"\nMetrikler: {out}")
