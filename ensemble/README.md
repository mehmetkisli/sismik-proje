# Multi-Seed Ensemble — v9 Mimarisi

> **Multi-seed ensemble** yaklaşımı ile v9 mimarisi (DeepLabV3+ + EfficientNet-B4 + 5-channel 2.5D + Lovász + xline-aware aug) farklı PyTorch random seed'leri ile bağımsız eğitilmiş, softmax tahminleri ortalanmıştır. **Ensemble v9'u Combined mIoU'da +1.41 puan geçti** ve tek-seed varyansını söndürdü.

**Tarih:** 2026-05-11/12

---

## 1. Motivasyon

Tek seed (42) ile yapılan v9 eğitiminin sonuçları **single-seed bias** riskine açıktı. Literatürde (Random effects nnU-Net 50-seed çalışması, Abid et al. 2022 sismik ensemble) multi-seed ensemble **tipik +1-2p mIoU** kazandırıyor + error bar ile dürüst raporlama sağlıyor.

**Hipotez:** SEED 42 + 43 + 44 ile 3 bağımsız v9 eğitimi → softmax averaging → daha yüksek + daha güvenilir Combined mIoU.

---

## 2. Konfigürasyon

- **Mimari:** v9 ile **birebir aynı** (DeepLabV3+ + EfficientNet-B4, 5-kanallı 2.5D, 384×384, QuadrupleLoss FP32 wrap)
- **Hyperparams:** v9 ile **birebir aynı** (batch 4 × accum 6, 100 epoch, patience 25, AdamW lr=1e-4 wd=1e-3, CosineAnnealingWarmRestarts T_0=25)
- **Sadece SEED değişkeni** (42, 43, 44) — `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `WeightedRandomSampler generator`
- **Ensemble inference:** Multi-scale TTA (HFlip + polarity + scale 0.75/1.0/1.25), her modelden softmax çıktısı → 3 modelin **ortalaması** → argmax

---

## 3. Sonuçlar (multi-scale TTA)

### Tek-seed sonuçları

| SEED | Best val | Combined mIoU | Test1 mIoU | Test2 mIoU | C4 Test2 IoU |
|---|---:|---:|---:|---:|---:|
| 42 (v9) | 0.8022 | 0.7769 | 0.7752 | **0.6869** ★ | **0.2304** ★ |
| 43 | 0.8271 | 0.7850 | 0.8041 | 0.6626 | 0.1560 |
| 44 | 0.8078 | **0.7920** ★ | **0.8095** ★ | 0.6727 | 0.1625 |

### Ensemble sonucu (softmax averaging, 3 model)

| Metrik | Ensemble | v9 (SEED 42) | Δ |
|---|---:|---:|---:|
| **Combined mIoU** | **0.7910** | 0.7769 | **+1.41p** ↑ |
| Test1 mIoU | 0.8043 | 0.7752 | **+2.91p** ↑ |
| Test2 mIoU | 0.6771 | 0.6869 | −0.98p ↓ |
| Class 4 Test2 | 0.1834 | 0.2304 | **−4.70p** ↓ |

### Seed varyansı

| | Combined mIoU |
|---|---|
| Mean | 0.7846 |
| Std | **±0.0062** (~±0.6 puan) |

---

## 4. 🎯 Kritik Akademik Bulgu

**v9 (SEED=42) Class 4 Test2 sonucu (0.2304) istatistiksel olarak outlier'dı.**

3 seed gerçek beklenen Class 4 Test2 değeri:
- SEED 42: 0.2304 (outlier-pozitif)
- SEED 43: 0.1560
- SEED 44: 0.1625
- **Mean: ~0.183 ± 0.04**

Bu, **tek-seed sonuçlarının güvenilirliği konusunda önemli bir uyarı**: v9'un 0.230 sayısını "Class 4 Test2'de büyük başarı" diye sunmak yanıltıcı olurdu. Multi-seed ensemble gerçek değeri (0.183) ortaya koydu — akademik dürüstlüğün net bir örneği.

**Sunum cümlesi:**
> "v9 modelinin Class 4 Test2 IoU'su 0.230 olarak tek seed ile rapor edilmişti. Ancak 3 bağımsız seed (42, 43, 44) ile yapılan multi-seed analizinde gerçek değerin 0.183 ± 0.04 olduğu görüldü; tek-seed sonuç istisnai/şanslı bir çıkıştı. Bu, segmentation modellerinin tek seed ile rapor edilmesinin riskinin kanıtıdır."

---

## 5. Hangi Metrikte Hangi Model Lider?

| Metrik | Lider | Değer |
|---|---|---|
| **Combined mIoU** | **SEED 44** | 0.7920 (ensemble 0.7910 ile denk) |
| Test1 mIoU | SEED 44 | 0.8095 |
| Test2 mIoU | SEED 42 (v9) | 0.6869 |
| Class 4 Test2 | SEED 42 (v9) | 0.2304 (outlier — ensemble 0.183 daha güvenilir) |
| Best val mIoU | SEED 43 | 0.8271 |

**Tez için raporlanacak ana sayı:**
> "Combined mIoU = 0.791 ± 0.006 (3-seed ensemble, multi-scale TTA, sade FP32 Lovász)"

---

## 6. Dosya Yapısı

```
ensemble/
├── README.md                    ← bu dosya
├── train_v9_seed.py             ← v9 mimarisini SEED parametresiyle eğitir
└── evaluate_ensemble.py         ← 3 best model softmax averaging inference
```

Çıktı dosyaları (`results/metrics/`):
- `v9_seed_43_metrics.json` — SEED 43 tek-seed sonuçları
- `v9_seed_44_metrics.json` — SEED 44 tek-seed sonuçları
- `v9_ensemble_metrics.json` — Ensemble + tüm tekil + variance bilgisi

Best model checkpoint'leri (`checkpoints_v7/`, gitignored):
- `deeplabv3plus_v9_best.pth` — SEED 42 (v9, mevcuttan)
- `v9_seed_43_best.pth` — SEED 43 best (343 MB)
- `v9_seed_44_best.pth` — SEED 44 best (343 MB)

---

## 7. Reproducibility

```bash
# SEED 43 + SEED 44 eğitimi (~3 saat toplam)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe ensemble/train_v9_seed.py 43
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe ensemble/train_v9_seed.py 44

# Ensemble inference (~15 dakika)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe ensemble/evaluate_ensemble.py
```

**Not:** SEED 42 ana v9 notebook'undan (`deeplabv3plus_v9.ipynb`) gelir. Yeniden üretmek için v9 notebook'unu çalıştırmak yeterli (best model `checkpoints_v7/deeplabv3plus_v9_best.pth` olarak kaydedilir).

---

## 8. Sonraki Adımlar

- **Yön 4 (Original-resolution evaluator):** Ensemble metriklerini Alaudah birebir protokolünde yeniden hesapla → "Alaudah baseline +X puan üstündeyiz" akademik iddia
- **Yön 1 (DTL loss):** Ensemble v2 — 3 seed'in QuintupleLoss (Lovász + DTL) ile yeniden eğitilmesi → Class 4 Test2 daha güçlü ele alınabilir

Detaylı analiz: `docs/PROJE_OZETI.md`
