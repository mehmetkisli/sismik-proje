"""Original-resolution v9 ensemble evaluator — Alaudah birebir kıyas için.

Fark (evaluate_ensemble.py'dan):
  - evaluate_ensemble.py: img + mask → 384x384 resize, metrikler 384x384'te
  - eval_originalres.py:  img → 384x384 (model), softmax → ORJİNAL (H, W) upsample,
                          metrikler ORJİNAL ÇÖZÜNÜRLÜKTE etiketlerle hesaplanır

Test1: axis=0 (inline) — her slice shape: (xline=701, depth=255)
Test2: axis=1 (xline)  — her slice shape: (inline=200, depth=255)

Çıktı:
  results/metrics/v9_ensemble_originalres_metrics.json
  + konsol karşılaştırma tablosu (resized vs original-res)

Kullanım:
  PYTHONIOENCODING=utf-8 conda run -n sismik python scripts/eval_originalres.py
  # veya tek bir checkpoint için:
  python scripts/eval_originalres.py --seeds 42
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
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="Hangi seedler kullanılsın (default 42 43 44)")
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
CKPT_DIR    = PROJECT_DIR / "checkpoints_v7"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATHS = {
    42: CKPT_DIR / "deeplabv3plus_v9_best.pth",
    43: CKPT_DIR / "v9_seed_43_best.pth",
    44: CKPT_DIR / "v9_seed_44_best.pth",
}

IMG_SIZE    = (384, 384)
N_NEIGHBORS = 2
N_CHANNELS  = 2 * N_NEIGHBORS + 1
NUM_CLASSES = 6
BATCH_SIZE  = args.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Cihaz: {device}  AMP: {use_amp}")


# ─── Veri ───────────────────────────────────────────────────────────────────
print("\nVeri yükleniyor...")
train_seismic = np.load(DATA_DIR / "train" / "train_seismic.npy")
test1_seismic = np.load(DATA_DIR / "test_once" / "test1_seismic.npy")
test1_labels  = np.load(DATA_DIR / "test_once" / "test1_labels.npy")
test2_seismic = np.load(DATA_DIR / "test_once" / "test2_seismic.npy")
test2_labels  = np.load(DATA_DIR / "test_once" / "test2_labels.npy")
print(f"  train_seismic: {train_seismic.shape}")
print(f"  test1_seismic: {test1_seismic.shape}  test1_labels: {test1_labels.shape}")
print(f"  test2_seismic: {test2_seismic.shape}  test2_labels: {test2_labels.shape}")

train_mean = train_seismic.mean(); train_std = train_seismic.std() + 1e-8
test1_norm = ((test1_seismic - train_mean) / train_std).astype(np.float32)
test2_norm = ((test2_seismic - train_mean) / train_std).astype(np.float32)
del train_seismic, test1_seismic, test2_seismic


class F3DatasetOrigRes(Dataset):
    """2.5D input → 384x384 (model için).  Mask ORJİNAL çözünürlükte döner.
    Çıktıda ayrıca (H_orig, W_orig) verir ki softmax inference sonrası geri upsample edilebilsin."""
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
        img = np.stack(slices, axis=-1)  # (H_orig, W_orig, C)
        mask = self.labels[idx] if self.axis == 0 else self.labels[:, idx]
        H_orig, W_orig = mask.shape

        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float()  # (C, H_orig, W_orig)
        img_t = F.interpolate(img_t.unsqueeze(0), size=IMG_SIZE, mode="bilinear",
                              align_corners=False).squeeze(0)  # (C, 384, 384)
        mask_t = torch.from_numpy(mask.copy().astype(np.int64))  # orijinal çözünürlükte
        return img_t, mask_t, H_orig, W_orig


def collate_fn(batch):
    """Batch içinde aynı (H, W) varsayımıyla."""
    imgs   = torch.stack([b[0] for b in batch], dim=0)
    masks  = torch.stack([b[1] for b in batch], dim=0)
    H_orig = batch[0][2]; W_orig = batch[0][3]
    return imgs, masks, H_orig, W_orig


test1_loader = DataLoader(F3DatasetOrigRes(test1_norm, test1_labels, axis=0),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                          collate_fn=collate_fn)
test2_loader = DataLoader(F3DatasetOrigRes(test2_norm, test2_labels, axis=1),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                          collate_fn=collate_fn)


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
    ious = [float(cm[c, c] / (cm[c, :].sum() + cm[:, c].sum() - cm[c, c] + 1e-8))
            for c in range(n)]
    dices = []
    for c in range(n):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        dices.append(float(2 * tp / (2 * tp + fp + fn + 1e-8)))
    # Frequency-Weighted IoU (Alaudah'ın raporladığı metrik)
    freqs = rs / rs.sum()
    fwiou = float(np.sum(freqs * np.array(ious)))
    return {
        "PA":   float((preds == targets).sum() / len(targets)),
        "MCA":  float(cacc.mean()),
        "per_class_acc": cacc.tolist(),
        "mIoU": float(np.mean(ious)),
        "per_class_iou": ious,
        "mean_dice": float(np.mean(dices)),
        "per_class_dice": dices,
        "FwIoU": fwiou,
    }


def run_model_tta_origres(model, loader):
    """Multi-scale TTA inference + softmax'i orijinal çözünürlüğe geri upsample.
    Dönüş: softmax list (her slice farklı H,W olabilir ama dataset axis sabit olduğu için aynı), targets list."""
    TTA_SCALES = [0.75, 1.25]
    all_softmax = []; all_targets = []
    model.eval()
    with torch.no_grad():
        for imgs, masks, H_orig, W_orig in loader:
            x = imgs.to(device, non_blocking=True)
            H_in, W_in = x.shape[2], x.shape[3]  # 384, 384

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

            # Multi-scale (model girişi 384x384 üzerinde)
            for sc in TTA_SCALES:
                hs = max(32, (int(H_in * sc) // 16) * 16)
                ws = max(32, (int(W_in * sc) // 16) * 16)
                xs = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    ls_ = model(xs)
                ls_up = F.interpolate(ls_, size=(H_in, W_in), mode="bilinear", align_corners=False)
                probs += torch.softmax(ls_up, 1); n_aug += 1
            probs /= n_aug  # (B, C, 384, 384)

            # ★ Softmax'i ORJİNAL çözünürlüğe upsample
            probs_orig = F.interpolate(probs, size=(H_orig, W_orig),
                                       mode="bilinear", align_corners=False)
            all_softmax.append(probs_orig.float().cpu().numpy())  # (B, C, H_orig, W_orig)
            all_targets.append(masks.numpy())  # (B, H_orig, W_orig)
    return np.concatenate(all_softmax, axis=0), np.concatenate(all_targets, axis=0)


# ─── Her seed için inference ──────────────────────────────────────────────
softmaxes_t1 = {}; softmaxes_t2 = {}; targets_t1 = None; targets_t2 = None
individual_results = {}

for seed in args.seeds:
    ckpt_path = CKPT_PATHS.get(seed)
    if ckpt_path is None or not ckpt_path.exists():
        print(f"\n⚠️ SEED {seed} checkpoint yok: {ckpt_path} — atlandı")
        continue
    print(f"\n[SEED {seed}] Model yükleniyor: {ckpt_path.name}")
    model = build_model()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    print(f"  Test1 inference (multi-scale TTA, original-res output)...")
    sm1, tg1 = run_model_tta_origres(model, test1_loader)
    print(f"    softmax shape: {sm1.shape}  targets shape: {tg1.shape}")
    print(f"  Test2 inference (multi-scale TTA, original-res output)...")
    sm2, tg2 = run_model_tta_origres(model, test2_loader)
    print(f"    softmax shape: {sm2.shape}  targets shape: {tg2.shape}")

    softmaxes_t1[seed] = sm1; softmaxes_t2[seed] = sm2
    if targets_t1 is None:
        targets_t1 = tg1; targets_t2 = tg2

    pred_t1 = sm1.argmax(1).reshape(-1); flat_t1 = tg1.reshape(-1)
    pred_t2 = sm2.argmax(1).reshape(-1); flat_t2 = tg2.reshape(-1)
    m1 = compute_metrics(pred_t1, flat_t1)
    m2 = compute_metrics(pred_t2, flat_t2)
    m_all = compute_metrics(np.concatenate([pred_t1, pred_t2]),
                            np.concatenate([flat_t1, flat_t2]))
    individual_results[seed] = {"test1": m1, "test2": m2, "combined": m_all}
    print(f"  SEED {seed} (original-res): "
          f"Combined mIoU={m_all['mIoU']*100:.2f}%  FwIoU={m_all['FwIoU']*100:.2f}%  "
          f"Test1={m1['mIoU']*100:.2f}%  Test2={m2['mIoU']*100:.2f}%  "
          f"C4 Test2 IoU={m2['per_class_iou'][4]*100:.2f}%")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ─── Ensemble: softmax averaging ─────────────────────────────────────────
if len(softmaxes_t1) >= 2:
    print(f"\n[ENSEMBLE] Softmax averaging ({len(softmaxes_t1)} model, original-res)...")
    ens_sm1 = np.mean(list(softmaxes_t1.values()), axis=0)
    ens_sm2 = np.mean(list(softmaxes_t2.values()), axis=0)

    ens_pred_t1 = ens_sm1.argmax(1).reshape(-1); flat_t1 = targets_t1.reshape(-1)
    ens_pred_t2 = ens_sm2.argmax(1).reshape(-1); flat_t2 = targets_t2.reshape(-1)
    ens_m1  = compute_metrics(ens_pred_t1, flat_t1)
    ens_m2  = compute_metrics(ens_pred_t2, flat_t2)
    ens_all = compute_metrics(np.concatenate([ens_pred_t1, ens_pred_t2]),
                              np.concatenate([flat_t1, flat_t2]))
else:
    ens_m1 = ens_m2 = ens_all = None


# ─── Resized (384x384) referans değerleri — karşılaştırma için ─────────
# Bu değerler results/metrics/v9_ensemble_metrics.json'dan kanonik
v9_ensemble_resized = {"combined_miou": 0.7910, "test1_miou": 0.7752,
                       "test2_miou": 0.6869, "c4_test2_iou": 0.183, "fwiou": 0.892}


print("\n" + "=" * 85)
print(" V9 ENSEMBLE — ORIGINAL-RESOLUTION EVALUATOR")
print("=" * 85)
if ens_all is not None:
    print(f"\n{'Metric':<22} {'Resized (384)':>15} {'Original-res':>15} {'Δ':>10}")
    print("-" * 85)
    rows = [
        ("Combined mIoU",  v9_ensemble_resized["combined_miou"], ens_all["mIoU"]),
        ("Combined FwIoU", v9_ensemble_resized["fwiou"],         ens_all["FwIoU"]),
        ("Test1 mIoU",     v9_ensemble_resized["test1_miou"],    ens_m1["mIoU"]),
        ("Test2 mIoU",     v9_ensemble_resized["test2_miou"],    ens_m2["mIoU"]),
        ("Class 4 Test2",  v9_ensemble_resized["c4_test2_iou"],  ens_m2["per_class_iou"][4]),
    ]
    for name, resz, orig in rows:
        delta = (orig - resz) * 100
        print(f"{name:<22} {resz*100:>13.2f}%  {orig*100:>13.2f}%  {delta:>+8.2f}p")
    print("=" * 85)

    CLS = ["Upper NS","Lower NS","Rijnland","Scruff","Zechstein","Under Zech"]
    print(f"\nPer-class IoU (Combined, original-res) — ENSEMBLE:")
    for i, n in enumerate(CLS):
        print(f"  S{i} {n:15s}: IoU={ens_all['per_class_iou'][i]:.4f}")


# ─── Kaydet ─────────────────────────────────────────────────────────────
results = {
    "model": f"v9 multi-seed ensemble (original-resolution evaluator, {len(softmaxes_t1)} models, "
             "softmax averaging, multi-scale TTA, softmax upsampled to original H×W before argmax)",
    "seeds": list(softmaxes_t1.keys()),
    "input_resolution": list(IMG_SIZE),
    "evaluation": "original resolution (Test1=701x255 inline-sliced, Test2=200x255 xline-sliced)",
    "individual": {str(s): r for s, r in individual_results.items()},
    "ensemble": {"test1": ens_m1, "test2": ens_m2, "combined": ens_all} if ens_all else None,
    "resized_reference": v9_ensemble_resized,
    "note": "Alaudah benchmark birebir kıyas için orijinal-çözünürlük değerleri kullanılmalı.",
}
out = METRICS_DIR / "v9_ensemble_originalres_metrics.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrikler: {out}")
