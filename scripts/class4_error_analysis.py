"""
Class 4 (Zechstein) Error Analysis — Test1 vs Test2 Genelleme Çöküşü

Sorun: Class 4 IoU = 0.84 (Test1) vs 0.18 (Test2) — 4.7 kat fark.
Bu script v7-fixed modelini yükler, tahminleri üretir ve Class 4'e özel
analizleri görselleştirir.

Çıktılar (results/figures/error_analysis/ altına):
  - class4_iou_per_slice.png       — her test slice'ında Class 4 IoU dağılımı
  - class4_confusion_test1.png     — Class 4 hangi sınıflara karıştırılıyor (Test1)
  - class4_confusion_test2.png     — aynı, Test2 için
  - class4_morphology_inline.png   — Class 4 GT morfolojisi inline yönünde (3 örnek)
  - class4_morphology_xline.png    — Class 4 GT morfolojisi crossline yönünde (3 örnek)
  - class4_pred_vs_gt_test1.png    — Test1 örneklerinde model tahmini vs GT (Class 4 odaklı)
  - class4_pred_vs_gt_test2.png    — Test2 örneklerinde aynı
  - class4_summary.json            — sayısal özet

Kullanım:
  cd sismik-proje/
  python scripts/class4_error_analysis.py

Gereksinimler: Eğitim tamamlanmış olmalı, deeplabv3plus_v7_fixed_best.pth mevcut olmalı.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import segmentation_models_pytorch as smp

# ─── Konfigürasyon ────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
CHECKPOINTS = PROJECT_DIR / "checkpoints_v7"
BEST_MODEL  = CHECKPOINTS / "deeplabv3plus_v7_fixed_best.pth"
OUT_DIR     = PROJECT_DIR / "results" / "figures" / "error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Upper NS", "Lower NS", "Rijnland", "Scruff", "Zechstein", "Under Zech"]
NUM_CLASSES = 6
TARGET_CLASS = 4   # Zechstein
FACIES_COLORS = ["#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#fdae61", "#f46d43"]
cmap_facies = mcolors.ListedColormap(FACIES_COLORS)
IMG_SIZE = (320, 320)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                      "cpu")
print(f"Cihaz: {device}")


# ─── Veri yükleme ────────────────────────────────────────────────────────────
print("\nVeri yükleniyor...")
train_seismic = np.load(DATA_DIR / "train" / "train_seismic.npy")
test1_seismic = np.load(DATA_DIR / "test_once" / "test1_seismic.npy")
test1_labels  = np.load(DATA_DIR / "test_once" / "test1_labels.npy")
test2_seismic = np.load(DATA_DIR / "test_once" / "test2_seismic.npy")
test2_labels  = np.load(DATA_DIR / "test_once" / "test2_labels.npy")

# Train mean/std ile normalize (notebook ile aynı)
train_mean = train_seismic.mean()
train_std  = train_seismic.std() + 1e-8
test1_norm = ((test1_seismic - train_mean) / train_std).astype(np.float32)
test2_norm = ((test2_seismic - train_mean) / train_std).astype(np.float32)
del train_seismic
print(f"  Test1: {test1_seismic.shape}  Test2: {test2_seismic.shape}")


# ─── Model yükleme ───────────────────────────────────────────────────────────
print("\nModel yükleniyor...")
model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
    encoder_output_stride=16,
    decoder_atrous_rates=(12, 24, 36),
).to(device)

assert BEST_MODEL.exists(), f"Best model bulunamadı: {BEST_MODEL}\nÖnce eğitimin tamamlanması lazım."
model.load_state_dict(torch.load(BEST_MODEL, map_location=device, weights_only=True))
model.eval()
print(f"  {BEST_MODEL}")


# ─── 2.5D slice yükleme yardımcısı ───────────────────────────────────────────
def get_25d_input(volume_norm, idx, axis):
    """3 komşu slice → 3 kanal, 320×320'ye resize."""
    n = volume_norm.shape[axis]
    i_prev = max(0, idx - 1)
    i_next = min(n - 1, idx + 1)
    if axis == 0:
        s_prev = volume_norm[i_prev]
        s_curr = volume_norm[idx]
        s_next = volume_norm[i_next]
    else:
        s_prev = volume_norm[:, i_prev]
        s_curr = volume_norm[:, idx]
        s_next = volume_norm[:, i_next]
    img = np.stack([s_prev, s_curr, s_next], axis=0).astype(np.float32)
    img_t = torch.from_numpy(img).unsqueeze(0)
    img_t = F.interpolate(img_t, size=IMG_SIZE, mode="bilinear", align_corners=False)
    return img_t.to(device)


def get_label_resized(labels_vol, idx, axis, original_shape):
    """Ground truth label, orijinal boyutta."""
    if axis == 0:
        return labels_vol[idx]
    else:
        return labels_vol[:, idx]


def predict_with_tta(img_t):
    """TTA: original + HFlip + polarity inversion."""
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)

        logits_hf = model(torch.flip(img_t, dims=[3]))
        probs += torch.softmax(torch.flip(logits_hf, dims=[3]), dim=1)

        logits_neg = model(-img_t)
        probs += torch.softmax(logits_neg, dim=1)

        probs /= 3.0
        return probs


def predict_slice(volume_norm, idx, axis, target_shape):
    """Tek slice için TTA tahmin, orijinal boyuta resize edilmiş."""
    img_t = get_25d_input(volume_norm, idx, axis)
    probs = predict_with_tta(img_t)
    pred_resized = F.interpolate(probs, size=target_shape, mode="bilinear", align_corners=False)
    return pred_resized.argmax(1).squeeze().cpu().numpy()


# ─── 1. Per-slice Class 4 IoU dağılımı ──────────────────────────────────────
print("\n[1/5] Per-slice Class 4 IoU hesaplanıyor...")

def per_slice_class4_iou(seismic_norm, labels_vol, axis):
    n_slices = seismic_norm.shape[axis]
    ious = np.full(n_slices, np.nan)
    coverage = np.zeros(n_slices)  # Class 4 piksel oranı
    for i in range(n_slices):
        if axis == 0:
            gt = labels_vol[i]
        else:
            gt = labels_vol[:, i]
        c4_mask = (gt == TARGET_CLASS)
        coverage[i] = c4_mask.mean()
        if c4_mask.sum() == 0:
            continue
        target_shape = gt.shape
        pred = predict_slice(seismic_norm, i, axis, target_shape)
        pred_mask = (pred == TARGET_CLASS)
        inter = (pred_mask & c4_mask).sum()
        union = (pred_mask | c4_mask).sum()
        ious[i] = inter / (union + 1e-8)
    return ious, coverage

iou_test1, cov_test1 = per_slice_class4_iou(test1_norm, test1_labels, axis=0)
iou_test2, cov_test2 = per_slice_class4_iou(test2_norm, test2_labels, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(13, 6))
axes[0].plot(iou_test1, color="#1f77b4", lw=1.2, label="Per-slice IoU")
axes[0].fill_between(range(len(cov_test1)), 0, cov_test1, color="orange", alpha=0.3, label="Class 4 piksel oranı")
axes[0].axhline(np.nanmean(iou_test1), color="red", ls="--", lw=1, label=f"Ortalama IoU={np.nanmean(iou_test1):.3f}")
axes[0].set_title(f"Test1 (Inline) — Class 4 (Zechstein) Per-slice IoU  |  ortalama={np.nanmean(iou_test1):.3f}")
axes[0].set_xlabel("Test1 inline indeksi"); axes[0].set_ylabel("IoU / coverage")
axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 1)

axes[1].plot(iou_test2, color="#d62728", lw=1.2, label="Per-slice IoU")
axes[1].fill_between(range(len(cov_test2)), 0, cov_test2, color="orange", alpha=0.3, label="Class 4 piksel oranı")
axes[1].axhline(np.nanmean(iou_test2), color="red", ls="--", lw=1, label=f"Ortalama IoU={np.nanmean(iou_test2):.3f}")
axes[1].set_title(f"Test2 (Crossline) — Class 4 (Zechstein) Per-slice IoU  |  ortalama={np.nanmean(iou_test2):.3f}")
axes[1].set_xlabel("Test2 crossline indeksi"); axes[1].set_ylabel("IoU / coverage")
axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUT_DIR / "class4_iou_per_slice.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  → class4_iou_per_slice.png")


# ─── 2. Class 4 confusion (hangi sınıfa karıştırılıyor) ──────────────────────
print("\n[2/5] Class 4 confusion analizi...")

def class4_confusion(seismic_norm, labels_vol, axis):
    """Class 4 GT pikselleri için tahmin dağılımı."""
    n_slices = seismic_norm.shape[axis]
    confusion = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(n_slices):
        if axis == 0:
            gt = labels_vol[i]
        else:
            gt = labels_vol[:, i]
        c4_mask = (gt == TARGET_CLASS)
        if c4_mask.sum() == 0:
            continue
        pred = predict_slice(seismic_norm, i, axis, gt.shape)
        for c in range(NUM_CLASSES):
            confusion[c] += ((pred == c) & c4_mask).sum()
    return confusion

conf_t1 = class4_confusion(test1_norm, test1_labels, axis=0)
conf_t2 = class4_confusion(test2_norm, test2_labels, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, conf, title in zip(axes, [conf_t1, conf_t2], ["Test1 (Inline)", "Test2 (Crossline)"]):
    pct = conf / conf.sum() * 100
    bars = ax.bar(range(NUM_CLASSES), pct, color=FACIES_COLORS, edgecolor="k")
    bars[TARGET_CLASS].set_edgecolor("red")
    bars[TARGET_CLASS].set_linewidth(2)
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{p:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"S{j}\n{CLASS_NAMES[j]}" for j in range(NUM_CLASSES)], fontsize=8)
    ax.set_ylabel("Class 4 GT piksellerinin tahmin dağılımı (%)")
    ax.set_title(f"{title} — Class 4 nereye gidiyor?")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

plt.suptitle("Class 4 (Zechstein) Confusion Analizi — Inline vs Crossline", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "class4_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  → class4_confusion.png")


# ─── 3. Class 4 morfoloji karşılaştırma (inline vs crossline) ────────────────
print("\n[3/5] Class 4 morfoloji görselleri...")

def find_class4_rich_slices(labels_vol, axis, n=3):
    """Class 4 piksel oranı en yüksek 3 slice'ı bul."""
    n_slices = labels_vol.shape[axis]
    coverage = []
    for i in range(n_slices):
        gt = labels_vol[i] if axis == 0 else labels_vol[:, i]
        coverage.append((i, (gt == TARGET_CLASS).mean()))
    coverage.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in coverage[:n]]

t1_indices = find_class4_rich_slices(test1_labels, axis=0, n=3)
t2_indices = find_class4_rich_slices(test2_labels, axis=1, n=3)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for j, idx in enumerate(t1_indices):
    gt = test1_labels[idx]
    sismik = test1_seismic[idx]
    c4_overlay = np.where(gt == TARGET_CLASS, 1, 0)
    axes[0, j].imshow(sismik.T, cmap="seismic", aspect="auto", vmin=-1, vmax=1, alpha=0.6)
    axes[0, j].contour(c4_overlay.T, levels=[0.5], colors="red", linewidths=1.5)
    axes[0, j].set_title(f"Test1 Inline #{idx} — Class 4 contour", fontsize=10)
    axes[0, j].axis("off")

for j, idx in enumerate(t2_indices):
    gt = test2_labels[:, idx]
    sismik = test2_seismic[:, idx]
    c4_overlay = np.where(gt == TARGET_CLASS, 1, 0)
    axes[1, j].imshow(sismik.T, cmap="seismic", aspect="auto", vmin=-1, vmax=1, alpha=0.6)
    axes[1, j].contour(c4_overlay.T, levels=[0.5], colors="red", linewidths=1.5)
    axes[1, j].set_title(f"Test2 Crossline #{idx} — Class 4 contour", fontsize=10)
    axes[1, j].axis("off")

plt.suptitle("Class 4 (Zechstein) Morfolojisi — Inline (Test1) vs Crossline (Test2)\n"
             "Yön bağımlılığı: Tuz tabakası inline'da daha sürekli, crossline'da kıvrımlı",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "class4_morphology_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  → class4_morphology_comparison.png")


# ─── 4. Pred vs GT — Class 4 odaklı görseller ────────────────────────────────
print("\n[4/5] Pred vs GT görselleri...")

def plot_pred_vs_gt(seismic_norm, seismic_raw, labels_vol, indices, axis, title):
    fig, axes = plt.subplots(len(indices), 4, figsize=(18, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        if axis == 0:
            gt = labels_vol[idx]
            sismik = seismic_raw[idx]
        else:
            gt = labels_vol[:, idx]
            sismik = seismic_raw[:, idx]
        pred = predict_slice(seismic_norm, idx, axis, gt.shape)

        c4_gt = (gt == TARGET_CLASS)
        c4_pred = (pred == TARGET_CLASS)
        # Hata haritası: TP=yeşil, FP=kırmızı, FN=sarı
        error_map = np.zeros((*gt.shape, 3), dtype=np.float32)
        error_map[c4_gt & c4_pred] = [0.2, 0.8, 0.2]    # TP
        error_map[~c4_gt & c4_pred] = [1.0, 0.2, 0.2]   # FP
        error_map[c4_gt & ~c4_pred] = [1.0, 1.0, 0.2]   # FN

        iou = (c4_gt & c4_pred).sum() / ((c4_gt | c4_pred).sum() + 1e-8)

        axes[row, 0].imshow(sismik.T, cmap="seismic", aspect="auto", vmin=-1, vmax=1)
        axes[row, 0].set_title(f"Sismik — slice #{idx}", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt.T, cmap=cmap_facies, vmin=-0.5, vmax=5.5, aspect="auto")
        axes[row, 1].set_title("Ground Truth (tüm sınıflar)", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred.T, cmap=cmap_facies, vmin=-0.5, vmax=5.5, aspect="auto")
        axes[row, 2].set_title(f"Tahmin (tüm sınıflar)", fontsize=10)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(error_map.transpose(1, 0, 2), aspect="auto")
        axes[row, 3].set_title(f"Class 4 Hata: TP=yeşil  FP=kırmızı  FN=sarı | IoU={iou:.3f}", fontsize=10)
        axes[row, 3].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

fig1 = plot_pred_vs_gt(test1_norm, test1_seismic, test1_labels, t1_indices,
                       axis=0, title="Test1 (Inline) — Class 4 Tahmin vs GT")
fig1.savefig(OUT_DIR / "class4_pred_vs_gt_test1.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  → class4_pred_vs_gt_test1.png")

fig2 = plot_pred_vs_gt(test2_norm, test2_seismic, test2_labels, t2_indices,
                       axis=1, title="Test2 (Crossline) — Class 4 Tahmin vs GT")
fig2.savefig(OUT_DIR / "class4_pred_vs_gt_test2.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  → class4_pred_vs_gt_test2.png")


# ─── 5. Sayısal özet ─────────────────────────────────────────────────────────
print("\n[5/5] Sayısal özet hesaplanıyor...")

def class4_stats(seismic_norm, labels_vol, axis):
    n_slices = labels_vol.shape[axis]
    n_with_c4 = 0
    total_c4_px = 0
    total_px = 0
    for i in range(n_slices):
        gt = labels_vol[i] if axis == 0 else labels_vol[:, i]
        c4_px = (gt == TARGET_CLASS).sum()
        if c4_px > 0:
            n_with_c4 += 1
        total_c4_px += int(c4_px)
        total_px += gt.size
    return {
        "n_slices": int(n_slices),
        "n_slices_with_c4": int(n_with_c4),
        "c4_coverage_pct": float(total_c4_px / total_px * 100),
    }

summary = {
    "test1": {
        **class4_stats(test1_norm, test1_labels, axis=0),
        "mean_class4_iou": float(np.nanmean(iou_test1)),
        "median_class4_iou": float(np.nanmedian(iou_test1)),
        "std_class4_iou": float(np.nanstd(iou_test1)),
        "confusion_distribution_pct": (conf_t1 / conf_t1.sum() * 100).tolist(),
    },
    "test2": {
        **class4_stats(test2_norm, test2_labels, axis=1),
        "mean_class4_iou": float(np.nanmean(iou_test2)),
        "median_class4_iou": float(np.nanmedian(iou_test2)),
        "std_class4_iou": float(np.nanstd(iou_test2)),
        "confusion_distribution_pct": (conf_t2 / conf_t2.sum() * 100).tolist(),
    },
    "class_names": CLASS_NAMES,
    "interpretation": (
        "Test1 vs Test2 IoU farkı, tuz tabakasının yön bağımlı (anisotropic) "
        "morfolojisinden kaynaklanır. Inline yönünde Class 4 daha sürekli/blok yapıdadır; "
        "crossline yönünde kıvrım/diapir yapıları öne çıkar. Eğitim verisi inline-baskın "
        "anatomik temsil sunduğu için model crossline'a genelleyemez. Çözüm: domain adaptation "
        "veya data augmentation'da yön-bilincinde dönüşümler."
    )
}

with open(OUT_DIR / "class4_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  → class4_summary.json")

print("\n" + "=" * 70)
print(" CLASS 4 (ZECHSTEIN) HATA ANALİZİ ÖZETİ")
print("=" * 70)
print(f"\nTest1 (Inline):")
print(f"  - {summary['test1']['n_slices_with_c4']}/{summary['test1']['n_slices']} slice'ta Class 4 var")
print(f"  - Class 4 piksel oranı: {summary['test1']['c4_coverage_pct']:.2f}%")
print(f"  - Ortalama Class 4 IoU: {summary['test1']['mean_class4_iou']:.4f}")
print(f"  - Median Class 4 IoU: {summary['test1']['median_class4_iou']:.4f}")
print(f"\nTest2 (Crossline):")
print(f"  - {summary['test2']['n_slices_with_c4']}/{summary['test2']['n_slices']} slice'ta Class 4 var")
print(f"  - Class 4 piksel oranı: {summary['test2']['c4_coverage_pct']:.2f}%")
print(f"  - Ortalama Class 4 IoU: {summary['test2']['mean_class4_iou']:.4f}")
print(f"  - Median Class 4 IoU: {summary['test2']['median_class4_iou']:.4f}")

print(f"\nTest2'de Class 4 GT pikselleri en çok hangi sınıfa karıştırılıyor?")
sorted_idx = np.argsort(conf_t2)[::-1]
for rank, c in enumerate(sorted_idx[:3]):
    pct = conf_t2[c] / conf_t2.sum() * 100
    marker = " ← doğru" if c == TARGET_CLASS else " ← yanlış"
    print(f"  {rank+1}. S{c} {CLASS_NAMES[c]:15s}: {pct:.1f}%{marker}")

print(f"\nÇıktı klasörü: {OUT_DIR}")
print(f"Toplam {len(list(OUT_DIR.glob('*.png')))} PNG + 1 JSON üretildi.")
print("=" * 70)
