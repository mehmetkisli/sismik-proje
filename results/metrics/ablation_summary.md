# Ablation Sonuçları — v9 Mimarisi

Baseline: **v9 single-seed (SEED=42)** — `deeplabv3plus_v9_metrics.json`

Multi-scale TTA + xline-aware aug + Lovász + 5-channel **ON** (tam pipeline).


## 1. Training Ablation (mimari/loss bileşenleri)

| Konfig | Loss | Ch | Xline-aware | Best val | Test1 | Test2 | **Combined** | Δ vs v9 | C4 Test2 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| v9 baseline (SEED=42) | Quadruple | 5 | ON | 80.22% | 77.52% | 68.69% | **77.69%** | +0.00p | 23.04% |
| Lovász OFF (TripleLoss) | Triple | 5 | ON | 79.76% | 77.67% | 68.63% | **77.92%** | +0.23p | 21.61% |
| 3-channel 2.5D (±1) | Quadruple | 3 | ON | 82.70% | 77.05% | 66.25% | **77.14%** | -0.55p | 11.99% |
| xline-aware aug OFF | Quadruple | 5 | OFF | 80.21% | 76.86% | 65.92% | **76.88%** | -0.81p | 13.16% |

## 2. TTA Varyant Ablation (inference-only)

⚠️ TTA ablation henüz çalıştırılmadı (`ablation/eval_tta_variants.py`).


## 3. Yorumlama Notları

- **Lovász katkısı:** TripleLoss baseline'a vs Quadruple delta = Lovász'ın net katkısı.
- **5-ch katkısı:** 3-channel vs 5-channel — özellikle C4 Test2'de fark öne çıkar (anisotropic morfoloji).
- **xline-aware aug katkısı:** OFF konfigi xline genelleme zayıflığını ölçer — Test2 mIoU farkı kritik.
- **TTA katkısı:** Multi-scale TTA, HFlip-only ve no-TTA arasındaki delta → inference-time augmentation faydası.
- Class 4 Test2 hassasiyeti her ablation'da ayrı raporlanmalı — single-seed varyans yüksek (std ~0.04).
