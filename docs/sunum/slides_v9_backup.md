---
marp: true
theme: default
size: 16:9
paginate: true
header: 'Yedek Slaytlar — Q&A için'
footer: 'DeepLabV3+ Sismik Fasiyes — Mehmet Kisli'
style: |
  section { font-size: 22px; padding: 50px; }
  h1 { color: #1a3a6c; font-size: 36px; }
  h2 { color: #1a3a6c; font-size: 28px; }
  table { font-size: 18px; }
  th { background-color: #1a3a6c; color: white; }
  .small { font-size: 16px; }
  .tag { background-color: #fff3cd; color: #856404; padding: 4px 10px; border-radius: 4px; font-size: 16px; }
---

<!-- _class: lead -->
# Yedek Slaytlar (Q&A)

**15 slayt — soru gelirse açılır, ana sunumda yer almaz.**

Her slaytın başında **[hangi soruda kullanılır]** etiketi var.

---

## YS-1 [S1, S2] — Methodology fix kod detayı

**Yol A: 3 satırlık fix**

```python
# cell-9 (split)
val_inline_idx = np.arange(160, 240)
BUFFER = 2
train_inline_idx = np.concatenate([
    np.arange(0, 160 - BUFFER),
    np.arange(240 + BUFFER, 401),
])

# cell-11 (Dataset)
def __getitem__(self, idx):
    if self.axis == 1:  # crossline
        img = volume[:, idx, :]
        # train_inline_mask: val piksellerini çıkar
        img = img[self.train_inline_mask]
    ...
```

Ek karmaşıklık: ~5 satır. Etki: 3 sızıntı tamamen kapatıldı.

---

## YS-2 [S3] — v3 → v9 tam evrim

| Sürüm | Combined mIoU | Anahtar değişiklik |
|---|---:|---|
| v3 (U-Net) | 0.4012 | Plain U-Net baseline |
| v5 | 0.7582 | + EffNet-B4 + DeepLabV3+ + 320 |
| v6 | 0.7580 | + 2.5D + WeightedSampler |
| v7-broken | 0.7926 | (leakage'lı yapay artış) |
| **v7-fixed** | **0.7668** | **+ Yol A methodology fix** |
| v7-c4fix | 0.7779 | + 5-ch + Lovász + xline-aware aug |
| v9 (single-seed) | 0.7769 | + 384×384 + multi-scale TTA |
| **v9 ensemble** | **0.7910 ± 0.006** | **+ 3-seed softmax averaging** |

**Net ilerleme:** v3 → v9 ensemble = **+39 puan mIoU**.

---

## YS-3 [S4] — Liu 2020 detayı (3D neden değil)

**Liu et al. 2020, Geophysics 10.1190/geo2019-0467.1:**

> "On the F3 facies dataset, section-based 2D models consistently outperform patch-based 3D approaches due to limited training volume."

**Pratik nedenler:**
- F3 = **tek 3D volüm** → 3D patch-based zorunlu → global jeolojik bağlam parçalanır
- Sismik için **3D pretrained encoder yok** (Med3D tıbbi, transfer kanıtlanmadı)
- VRAM patlaması: 16-24 GB+ gerekir, bizim 3060 Ti 8 GB

**Bizim 2.5D = ImageNet uyumlu, VRAM-dostu, jeolojik bağlam korunur.**

---

## YS-4 [S5] — Sınırlı ablation tablo (tek-seed, SEED=42)

| Konfig | Combined mIoU | Δ vs v9 | Class 4 Test2 | Δ C4 | Yorum |
|---|---:|---:|---:|---:|---|
| **v9 (full pipeline)** | **0.7769** | — | **0.2304** | — | baseline |
| − Lovász (Triple loss) | 0.7792 | +0.23p | 0.2161 | **−1.43p** | Lovász Class 4 minörlüğüne katkı |
| − 5-ch (3-ch 2.5D, ±1) | 0.7714 | −0.55p | 0.1199 | **−11.05p** ⚠️ | 2.5D context Zechstein için kritik |
| − xline-aware aug | 0.7688 | −0.81p | 0.1316 | **−9.88p** ⚠️ | xline aug C4+Test2 için kritik |
| − TTA varyantları | — | — | — | — | (tezin uzun versiyonunda) |
| − Mixup / Focal alpha | — | — | — | — | (tezin uzun versiyonunda) |

**Hikâye:** 5-channel 2.5D + xline-aware aug, Class 4 Test2 için **olmazsa olmaz** — her birinin OFF konfigi ~−10 puan etki yaratıyor. Lovász Combined'a değil, **doğrudan minör sınıfa** katkı sağlıyor.

<span class="tag">Tek-seed ablation (SEED=42). Multi-seed varyans Class 4 için ±4p std — bulgular ana çizgi göstergesidir.</span>

---

## YS-5 [S6] — Class 4 hata mode analizi (Test2)

**Class 4 GT piksellerinin hangi sınıfa atandığı:**

| Atanan Sınıf | Yüzde |
|---|---:|
| **C4 Zechstein (doğru)** | %26 |
| C5 Under Zechstein (komşu alt) | %59 |
| C3 Scruff (komşu üst) | %12 |
| Diğer | %3 |

**Yorum:** Model Zechstein'i **tamamen kaybetmiyor**, sınırını yanlış çiziyor. Anisotropic morfoloji (inline blok ↔ xline diapir) genelleme zafiyeti.

→ Bu, mimari sorun değil **alan adaptasyonu** problemi.

---

## YS-6 [S7] — Tam SOTA literatür tablosu

| Yöntem | Yıl | Sayı | Split | Karşılaştırılabilir? |
|---|---|---|---|---|
| Alaudah section + aug + skip | 2019 | FwIoU 0.832, PA 0.905, MCA 0.817 | Geographic | ✅ Benchmark referansı |
| Liu et al. (3D CNN + GAN) | 2020 | Paywall, sayı yok | Geographic | ⚠️ |
| Wiley/Hindawi ensemble | 2022 | mIoU 0.9392 | Random patch, 7 sınıf | ❌ |
| CONSS semi-sup | 2023 | mIoU 0.9462 | Kendi split | ❌ |
| AdaSemSeg target-only | 2025 | FwIoU 0.86 | Farklı F3 split | ⚠️ |
| UmixClick interactive | 2025 | mIoU 0.7666, PA 0.9351 | Belirsiz | ❌ Interactive |
| **v9 ensemble (bizim)** | **2026** | **mIoU 0.7910 ± 0.006, PA 0.940, MCA 0.873** | **Alaudah geographic** | **✅** |

**Detay:** [`docs/literature_table.md`](../literature_table.md)

---

## YS-7 [S8] — K-fold CV maliyet analizi

**5-fold CV, v9 mimarisi için:**

| Fold | Eğitim süresi | GPU saat |
|---|---|---|
| Fold 1 | ~8 saat | 8 |
| Fold 2 | ~8 saat | 8 |
| Fold 3 | ~8 saat | 8 |
| Fold 4 | ~8 saat | 8 |
| Fold 5 | ~8 saat | 8 |
| **Toplam** | **~40 saat** | **40** |

**Multi-seed (3 seed) maliyeti:** ~24 saat. Multi-seed varyans ölçümünde benzer rol oynar, **maliyet 1.7x daha düşük**.

**Sonuç:** Yüksek lisans semineri için multi-seed pragmatik seçim. Tezin uzun versiyonunda K-fold planlanıyor.

---

## YS-8 [S9, S11] — SFM tam deneysel kurulum

**SFM v1 (Sheng 2024 base):**
- Backbone: **ViT-B/16**, MAE pretrained on 2.3M sismik slice
- Input: 5-ch 2.5D → 1-ch projection (Sheng'in beklediği format)
- Head: MLAHead (multi-layer aggregation)
- Encoder lr: **5e-5** (frozen-ish), decoder lr: 1e-3
- Sonuç: Combined 0.7281, **Class 4 Test2 = 0.270** ← lider

**SFM v2 (agresif fine-tune):**
- Encoder lr: **2e-4** + layer-wise lr decay 0.9
- WeightedRandomSampler rare classes ×15
- Sonuç: Combined 0.7677, **best val 0.823** ← lider, ama Class 4 Test2 0.20'ye düştü

**Ders:** Encoder agresif fine-tune Test1'i iyileştirir, Test2 genellemesini bozar. Trade-off.

---

## YS-9 [S10] — v9 vs v7-c4fix per-class breakdown

| Sınıf | v7-c4fix | v9 (single) | v9 ensemble | Δ ensemble |
|---|---:|---:|---:|---:|
| C0 Upper NS | 0.943 | 0.947 | **0.951** | +0.008 |
| C1 Lower NS | 0.791 | 0.797 | **0.812** | +0.021 |
| C2 Rijnland | 0.927 | 0.936 | **0.941** | +0.014 |
| C3 Scruff | 0.621 | 0.601 | **0.629** | +0.008 |
| **C4 Zechstein** | 0.789 | **0.784** | **0.803** | +0.014 |
| C5 Under Zech | 0.588 | 0.601 | **0.609** | +0.021 |

**Ensemble net katkı:** her sınıfta +0.01–0.02p. C5'te %3.5p iyileşme — küçük sınıflar ensemble'dan en çok faydalanıyor.

---

## YS-10 [S13] — 3-seed std her sınıf için

| Sınıf (Combined) | Mean IoU | Std | CV (std/mean) |
|---|---:|---:|---:|
| C0 Upper NS | 0.947 | 0.005 | 0.5% |
| C1 Lower NS | 0.808 | 0.010 | 1.2% |
| C2 Rijnland | 0.938 | 0.004 | 0.4% |
| C3 Scruff | 0.625 | 0.028 | 4.5% |
| **C4 Zechstein** | **0.797** | **0.013** | **1.7%** |
| C5 Under Zech | 0.603 | 0.018 | 3.0% |

**Class 4 Test2 detayı:** 0.230 / 0.156 / 0.163 → mean 0.183, **std 0.041** (CV %22 — yüksek!)

→ Class 4 Test2'de seed varyansı, modelin instabilitesinin somut göstergesi.

---

## YS-11 [S12] — Resize evaluator etki tahmini

**Eval çözünürlüğünün metriklere etkisi (deneysel tahmin):**

| Metrik | Resize hassasiyeti | Tahmini fark (384 vs orijinal) |
|---|---|---|
| Pixel Accuracy | Düşük | ±0.5 puan |
| Mean Class Acc | Düşük | ±1.0 puan |
| mIoU / FwIoU | **Yüksek** | **±2–4 puan** (öngörülmesi zor) |

**Sebep:** Resize sırasında **küçük sınıfların sınır pikselleri yumuşatılır** — IoU'da büyük etki, accuracy'de küçük.

**Beklenti:** Orijinal-resolution evaluator yapıldığında mIoU bir miktar **düşebilir** (resize sınır pikselleri görmezden geliyor). Bunu da dürüstçe söylüyoruz.

---

## YS-12 [S14] — AMP + Lovász FP32 wrap kod

```python
from torch.cuda.amp import autocast

with autocast():  # global fp16 context
    logits = model(x)                          # fp16
    loss_ce = ls_ce(logits, y)                 # fp16
    loss_dice = dice(logits, y)                # fp16
    loss_focal = focal(logits, y)              # fp16

    with autocast(enabled=False):              # FP32 wrap
        logits_fp32 = logits.float()
        loss_lov = lovasz_softmax(logits_fp32, y)

    total = 0.30 * loss_ce + 0.25 * loss_dice \
          + 0.25 * loss_focal + 0.20 * loss_lov

scaler.scale(total).backward()
```

**Standart PyTorch pattern** — `autocast(enabled=False)` ile precision karıştırma. NaN sorunu çözüldü, gradient scaler bozulmadı.

---

## YS-13 [misc] — Hyperparameter tablosu

| Parametre | Değer | Notlar |
|---|---|---|
| Optimizer | AdamW | β1=0.9, β2=0.999 |
| Learning rate | 1e-4 | base lr |
| Weight decay | 1e-3 | |
| Scheduler | CosineAnnealingWarmRestarts | T_0=25, eta_min=1e-6 |
| Batch size | 4 | gradient accum=6 → eff. batch 24 |
| Epochs | 100 | early stopping patience=25 |
| Image size | 384×384 | |
| Loss weights | 0.30/0.25/0.25/0.20 | LS-CE/Dice/Focal/Lovász |
| Focal γ | 2.0 | |
| LS ε | 0.1 | |
| Mixup α | 0.2 | p=0.5 |
| Sampler boost | C4/C5 ×10 | WeightedRandomSampler |

**Donanım:** RTX 3060 Ti (8 GB VRAM), AMP, gradient accum.

---

## YS-14 [misc] — VerticalFlip neden yok

**Sismikte derinlik ekseni jeofiziksel olarak ters çevrilemez:**

- Yer çekimi → üst tabakalar üstte (genç sedimanlar), alt tabakalar altta (yaşlı)
- VerticalFlip → "yaşlı kayalar üstte" gibi imkansız bir konfigürasyon üretir
- Model bu yanlış görüntüler üzerinde eğitilirse stratigrafik sıralamayı bozar

**Bizim aug listesi:**
- ✅ HorizontalFlip (yatay simetri jeolojik olarak geçerli)
- ✅ ShiftScaleRotate (küçük açı)
- ✅ ElasticTransform, GridDistortion (xline-aware)
- ✅ Polarity inversion (sismik sinyal × −1, ters reflektörler)
- ❌ **VerticalFlip** (jeolojik kural ihlali)

---

## YS-15 [Future Work] — Heterojen ensemble deney planı

**Hipotez:** v9 + SFM v1 softmax averaging → en iyisi.

| Model | Combined | C4 Test2 | Hangi konuda iyi |
|---|---:|---:|---|
| v9 ensemble | 0.791 | 0.183 | Genel performans |
| SFM v1 | 0.728 | 0.270 | Class 4 morfoloji |
| **Heterojen ensemble (öngörü)** | **~0.79** | **~0.24-0.27** | İkisinin gücü |

**Yöntem:**
1. v9 ensemble softmax çıktısı (mevcut)
2. SFM v1 softmax çıktısı (mevcut)
3. Ağırlıklı ortalama: `0.5 · softmax_v9 + 0.5 · softmax_sfm` (veya öğrenilebilir w)
4. argmax → final tahmin

**Maliyet:** Sadece inference, ~30 dk GPU. **Yüksek değer / düşük maliyet** future work.

---

<!-- _class: lead -->
# Yedek Slaytlar Bitti

Ana sunuma dönmek için **Esc** veya slayt 1.

**Hatırlatma:** Her yedek slaydın başındaki `[Sx]` etiketi, hangi soru için olduğunu gösterir. Provada ezberle.
