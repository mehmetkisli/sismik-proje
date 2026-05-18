"""Compute FwIoU directly from ensemble per-class IoU + GT label frequencies.

Tezdeki FwIoU iddiası (0.892) yeniden hesaplanır — frekanslar GT etiketlerinden
(test1_labels.npy + test2_labels.npy), per-class IoU değerleri ensemble JSON'dan.

Çıktı: stdout + results/metrics/fwiou_ensemble_recompute.json
"""
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
DATA = PROJECT / "data" / "test_once"
METRICS = PROJECT / "results" / "metrics"

# Load GT labels
t1 = np.load(DATA / "test1_labels.npy")
t2 = np.load(DATA / "test2_labels.npy")

NUM_CLASSES = 6

# Per-class pixel counts
t1_counts = np.array([(t1 == c).sum() for c in range(NUM_CLASSES)], dtype=np.int64)
t2_counts = np.array([(t2 == c).sum() for c in range(NUM_CLASSES)], dtype=np.int64)
combined_counts = t1_counts + t2_counts

t1_freq = t1_counts / t1_counts.sum()
t2_freq = t2_counts / t2_counts.sum()
combined_freq = combined_counts / combined_counts.sum()

# Load ensemble per-class IoU
with open(METRICS / "v9_ensemble_metrics.json", "r", encoding="utf-8") as f:
    ens = json.load(f)

ens_iou_t1 = np.array(ens["ensemble"]["test1"]["per_class_iou"])
ens_iou_t2 = np.array(ens["ensemble"]["test2"]["per_class_iou"])
ens_iou_combined = np.array(ens["ensemble"]["combined"]["per_class_iou"])

# FwIoU = sum(freq_c * IoU_c)
fwiou_t1 = float((t1_freq * ens_iou_t1).sum())
fwiou_t2 = float((t2_freq * ens_iou_t2).sum())
fwiou_combined = float((combined_freq * ens_iou_combined).sum())

# Comparison with old thesis number (used SEED 42 confusion-matrix frequencies)
# Old reported: 0.892 combined
print("=" * 70)
print("FwIoU Yeniden Hesaplama — GT frekansları + Ensemble per-class IoU")
print("=" * 70)
print()
print("GT pikselleri (test-time):")
print(f"  Test1 toplam:    {t1_counts.sum():>12,}")
print(f"  Test2 toplam:    {t2_counts.sum():>12,}")
print(f"  Combined toplam: {combined_counts.sum():>12,}")
print()

class_names = ["S0 Upper NS", "S1 Lower NS", "S2 Rijnland",
               "S3 Scruff", "S4 Zechstein", "S5 Under Zech"]
print(f"{'Sınıf':<15} {'T1 freq':>9} {'T2 freq':>9} {'Comb freq':>10}"
      f" {'Ens IoU (Comb)':>15}")
print("-" * 70)
for i, name in enumerate(class_names):
    print(f"{name:<15} {t1_freq[i]:>9.4f} {t2_freq[i]:>9.4f}"
          f" {combined_freq[i]:>10.4f} {ens_iou_combined[i]:>15.4f}")

print()
print("=" * 70)
print("FwIoU sonuçları:")
print(f"  Test1:    {fwiou_t1:.4f}")
print(f"  Test2:    {fwiou_t2:.4f}")
print(f"  Combined: {fwiou_combined:.4f}  (tezdeki eski sayı: 0.892)")
print("=" * 70)

# Save
out = {
    "method": "ensemble_per_class_iou * gt_label_frequencies",
    "fwiou": {
        "test1": fwiou_t1,
        "test2": fwiou_t2,
        "combined": fwiou_combined,
    },
    "gt_frequencies": {
        "test1": t1_freq.tolist(),
        "test2": t2_freq.tolist(),
        "combined": combined_freq.tolist(),
    },
    "ensemble_per_class_iou": {
        "test1": ens_iou_t1.tolist(),
        "test2": ens_iou_t2.tolist(),
        "combined": ens_iou_combined.tolist(),
    },
    "alaudah_baseline_combined": 0.832,
    "delta_vs_alaudah": fwiou_combined - 0.832,
}
out_path = METRICS / "fwiou_ensemble_recompute.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(f"\nKaydedildi: {out_path}")
