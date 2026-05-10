# Derin Öğrenme ile Sismik Fasiyes Sınıflandırması
## Proje Planı — DeepLabV3+ Tabanlı Segmentasyon (v7-fixed)

**Ders:** Bilgisayarda Görme (Yüksek Lisans, Bahar 2026)
**Veri Seti:** Netherlands F3 Block — Alaudah et al. (2019)
**Mimari:** DeepLabV3+ + EfficientNet-B4 (2.5D Multi-View)
**Ortam:** RTX 3060 Ti (8 GB VRAM, AMP)

> **NOT:** Bu plan ilk yazıldığında U-Net + 256×256 + 50 epoch tasarımı içeriyordu. v3 → v5 → v6 → v7 evrimi sonucunda mimari ve hiperparametreler değişti. Bu güncel sürüm aktif `deeplabv3plus_v7.ipynb` dosyasının spesifikasyonudur. Eski U-Net tasarımı `archive/` klasörü dışında artık kullanılmıyor.

---

## 1. Problemin Tanımı

Sismik fasiyes sınıflandırması, yeraltı jeolojik yapılarının sismik dalga yansıma örüntülerine göre kategorilere ayrılması işlemidir. Geleneksel yöntemde jeofizikçiler bu yorumu manuel olarak yapar; bu hem zaman alıcıdır hem de yorumcu tutarlılığı sınırlıdır.

Bu projede **DeepLabV3+** mimarisi kullanılarak sismik kesitler piksel bazında (6 fasiyes sınıfı) otomatik olarak sınıflandırılır.

**6 fasiyes sınıfı:**

| Sınıf | İsim | Piksel Sayısı (Train) | Oran |
|-------|------|----------------------|------|
| 0 | Upper North Sea | 20.137.839 | %28.09 |
| 1 | Lower North Sea | 8.519.666 | %11.89 |
| 2 | Rijnland | 34.831.122 | %48.59 |
| 3 | Scruff | 4.760.778 | %6.64 |
| 4 | Zechstein | 2.350.150 | %3.28 |
| 5 | Under Zechstein | 1.081.200 | %1.51 |

**Temel zorluk:** Sınıf 2 tüm piksellerin %48'ini, Sınıf 5 yalnızca %1.5'ini oluşturmaktadır. Bu dengesizlik modelin küçük sınıfları görmezden gelmesine yol açar — Focal Loss, weighted sampling ve rare-class boost ile ele alınmıştır.

---

## 2. Veri Seti

### 2.1 Kaynak

- **Benchmark:** Alaudah, Y. et al. (2019). "A Machine Learning Benchmark for Facies Classification." *Interpretation*, 7(3).
- **İndirme:** Zenodo `10.5281/zenodo.3755060` (~1 GB)
- **Lisans:** CC-BY-SA 3.0

### 2.2 Veri Yapısı

```
data/
├── train/
│   ├── train_seismic.npy     → (401, 701, 255)  float64   ~547 MB
│   └── train_labels.npy      → (401, 701, 255)  uint8     ~68 MB
└── test_once/
    ├── test1_seismic.npy     → (200, 701, 255)  float64   ~273 MB
    ├── test1_labels.npy      → (200, 701, 255)  uint8     ~34 MB
    ├── test2_seismic.npy     → (601, 200, 255)  float64   ~234 MB
    └── test2_labels.npy      → (601, 200, 255)  uint8     ~29 MB
```

**Boyut:** `(inline sayısı, crossline sayısı, derinlik)`. Sismik değerler [-1, 1] aralığına önceden normalize edilmiş.

### 2.3 Train / Validation / Test Bölmesi (v7-fixed Methodology)

```
Train inline:  [0, 158) ∪ [242, 401)  →  317 slice (iki blok, ±2 buffer)
Val inline:    [160, 240)              →  80 slice (orta blok)
Train xline:   [0, 601)                →  601 slice (val pikselleri image içinden cropped)
Test1:         200 inline              →  ayrı volume, final değerlendirme
Test2:         601 crossline           →  ayrı volume, generalization testi
```

**Methodology fix gerekçesi (Yol A, 2026-05-09 commit `0955d57`):**
- Eski split son %20'yi val olarak ayırıyordu → lokasyon bias riski
- Eski xline training tüm 601 crossline'ı kullanıyordu → val inline pikselleri her xline image'ında bulunduğu için 3D leakage
- ±2 inline buffer → 2.5D komşu sızıntısı engeli
- Detay: `docs/PROJE_OZETI.md` Bölüm 7

**Kritik kural:** Test setleri eğitim boyunca hiç kullanılmaz. Yalnızca eğitim tamamlandıktan sonra TTA ile bir kez çalıştırılır.

---

## 3. Teknik Pipeline

### 3.1 Genel Akış

```
Veri İndirme (Zenodo) → EDA → Methodology-Fixed Split → 2.5D Dataset
    → Volume Normalize → Train/Val/Test DataLoaders (WeightedRandomSampler)
    → DeepLabV3+ Model → Triple Loss + Mixup → AdamW + CosineWR
    → Training Loop (100 epoch, AMP, grad accum, early stop)
    → Best Model Save → TTA Evaluation → Görselleştirmeler + Versiyon Karşılaştırma
```

### 3.2 Keşifsel Veri Analizi (EDA)

- Sınıf piksel dağılımı çubuk grafiği
- Inline + crossline kesit örnekleri (sismik + ground-truth yan yana)
- Class frequency hesaplama → Focal alpha tablosu

### 3.3 Dataset ve DataLoader

**Her örnek (2.5D):**
- Girdi: 3 komşu slice → 3 kanallı görüntü, 320×320'ye resize → shape `(3, 320, 320)`
- Etiket: Merkez slice'ın piksel etiketleri → shape `(320, 320)`, her piksel 0-5

**2.5D mantığı:** Her slice için `[i-1, i, i+1]` slice'ları stacklenip RGB-benzeri bir görüntü oluşturulur. ImageNet pretrained encoder doğrudan uyumludur.

**DataLoader:**
```python
BATCH_SIZE  = 6      # 8 GB VRAM için sınırda; ACCUM_STEPS=4 ile efektif batch 24
NUM_WORKERS = 2
sampler     = WeightedRandomSampler(...)  # rare-class boost
```

### 3.4 Mimari — DeepLabV3+ + EfficientNet-B4

```python
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=6,
    encoder_output_stride=16,
    decoder_atrous_rates=(12, 24, 36),
)
```

**Toplam parametre:** ~24M
**Mimari mantık:** ASPP (Atrous Spatial Pyramid Pooling) ile çoklu ölçek bağlamı yakalanır — sismik tabakaların farklı kalınlıklarına uygun.

**3D yerine 2.5D tercih nedeni:** Tek volume eğitim verisi → 3D'de patch-based zorunlu, global jeolojik bağlam parçalanır. Ayrıca sismik için 3D pretrained encoder yok. Liu et al. 2020 (Geophysics) F3'te section-based 2D > patch-based 3D olduğunu göstermiştir. Detay: `docs/PROJE_OZETI.md` Bölüm 8.

### 3.5 Loss Fonksiyonu — Triple Loss

```
Loss = 0.4 · LabelSmoothingCE(eps=0.1) + 0.3 · DiceLoss + 0.3 · FocalLoss(γ=2)
```

- **LS-CE:** Aşırı güven (overconfidence) önler, kalibrasyonu iyileştirir
- **Dice:** Örtüşme oranını doğrudan optimize eder; sınıf dengesizliğine az duyarlı
- **Focal (γ=2, α=class_freq inverse):** Zor örneklere odaklanır, azınlık sınıflar için kritik

**Mixup augmentation:** α=0.2, p=0.5 per batch — regularization ve genelleme için.

### 3.6 Optimizer ve Scheduler

```python
optimizer = AdamW(lr=1e-4, weight_decay=1e-3)
scheduler = CosineAnnealingWarmRestarts(T_0=25, T_mult=1, eta_min=1e-6)
```

CosineAnnealingWarmRestarts ile her 25 epoch'ta lr resetlenir — local minima'dan çıkış için.

### 3.7 Training Loop

- 100 epoch, AMP (mixed precision), gradient accumulation ×4
- Early stopping patience=25 (val mIoU plateau)
- Checkpoint resume desteği var
- Her epoch sonu: train_loss, val_loss, val_mIoU, val_dice, lr kaydedilir

### 3.8 Augmentation

- HorizontalFlip (p=0.5)
- ShiftScaleRotate (rotate ±10°, p=0.5)
- ElasticTransform (α=80, σ=10, p=0.3)
- GridDistortion (p=0.3)
- RandomBrightnessContrast (p=0.4)
- GaussNoise (p=0.3)
- CoarseDropout (p=0.2)
- Polarity inversion (p=0.5) — sismik fizik
- **VerticalFlip YOK** — derinlik ekseni jeofizik olarak ters çevrilemez

### 3.9 Test Time Augmentation (TTA)

Test sırasında 3 transformasyon: orijinal + HFlip + polarity inversion → softmax çıktıları averaging.

### 3.10 Değerlendirme Metrikleri

| Metrik | Formül | Açıklama |
|--------|--------|----------|
| IoU (Jaccard) | TP / (TP + FP + FN) | Ana segmentasyon metriği |
| **mIoU** | ortalama IoU | Genel başarı (sınıflar arası eşit ağırlıklı) |
| Dice / F1 | 2·TP / (2·TP + FP + FN) | Daha hassas |
| Pixel Accuracy (PA) | doğru piksel / toplam | Yanıltıcı (S2 baskın) |
| Mean Class Accuracy (MCA) | ortalama class recall | Sınıf dengesizliğinde önemli |

**Test protokolü:**
1. Eğitim bitince `deeplabv3plus_v7_fixed_best.pth` yükle
2. Test1 üzerinde TTA ile değerlendir (inline yönü)
3. Test2 üzerinde TTA ile değerlendir (crossline — generalization)
4. Combined metrik: Test1 + Test2 birleşik
5. Per-class IoU/Dice raporla (özellikle azınlık sınıfları için)
6. SOTA literatür ile karşılaştır (`docs/literature_table.md` — derleme aşamasında)

### 3.11 Görselleştirme

- Eğitim eğrileri (train/val loss + val mIoU + lr)
- Confusion matrix (raw + normalize)
- Segmentasyon karşılaştırması (sismik | GT | tahmin) — Test1 ve Test2 örnekleri
- Per-class IoU/Dice bar chart
- Versiyon karşılaştırma tablosu (v3 / v5 / v7-broken / v7-fixed)

---

## 4. Mevcut Sürüm: v7-fixed

```
deeplabv3plus_v7.ipynb (33 hücre)
├── ✅ Veri yükleme + EDA
├── ✅ Methodology-fixed train/val split (Yol A)
├── ✅ 2.5D Dataset + train_inline_mask cropping
├── ✅ WeightedRandomSampler (rare-class boost)
├── ✅ DeepLabV3+ + EfficientNet-B4
├── ✅ Triple Loss + Mixup + AdamW + CosineWR
├── ✅ Training loop (100 epoch + AMP + grad accum + early stop + checkpoint resume)
├── ✅ TTA evaluation
├── ✅ Görselleştirmeler + versiyon karşılaştırma
└── 🚂 Eğitim devam ediyor — yeni sayılar `results/metrics/deeplabv3plus_v7_fixed_metrics.json`
```

**Eski (v7-broken) sayılar (methodology fix öncesi):**
| Metrik | Test1 | Test2 | Combined |
|---|---|---|---|
| mIoU | 0.7881 | 0.6986 | 0.7926 |
| mean Dice | 0.8726 | 0.7859 | 0.8775 |
| PA | 0.9356 | 0.9469 | 0.9413 |
| **Best val mIoU** | — | — | **0.6617** (paradoksal düşük) |

---

## 5. Yapılacaklar (2026-05-09 sonrası)

### Şu anda (eğitim devam ederken — paralel)
- [x] Methodology fix Yol A uygulandı
- [x] PROJE_PLANI.md güncellendi (bu dosya)
- [ ] SOTA literatür karşılaştırma tablosu (`docs/literature_table.md`)
- [ ] Limitations bölümü akademik metni
- [ ] Class 4 Test2 error analysis kodu

### Eğitim biter bitmez
- [ ] v7-fixed sayılarını yorumla; "öncesi vs sonrası" tablosu üret
- [ ] Class 4 Zechstein Test2 IoU=0.18 felaketi için hata analizi (inline vs crossline morfoloji karşılaştırma)

### Hafta 2-3 (eğer vakit kalırsa)
- [ ] Multi-seed (42/43/44) → mean±std
- [ ] Mini ablation (TTA-off, Mixup-off, single-channel)
- [ ] Architecture ablation (Unet++, MAnet) — methodology fix üzerinde

### Sunum hazırlığı (Hafta 4-5)
- [ ] 40 dakikalık slayt yapısı
- [ ] Limitations slaytı (val split bias, leakage öncesi/sonrası, single seed)
- [ ] Future work slaytı (ThinkOnward GFM fine-tune, domain adaptation)
- [ ] Beklenen zorlu sorular için cevap notları

---

## 6. Zaman Çizelgesi (2026-05-09 itibarıyla)

```
9-15 Mayıs    Hafta 1: Methodology fix retrain + SOTA tablosu + Limitations
16-22 Mayıs   Hafta 2: Multi-seed + ablation + Class 4 error analysis
23-29 Mayıs   Hafta 3: Architecture ablation veya GFM fine-tune (karar)
30 May-5 Haz  Hafta 4: Sunum hazırlığı, slaytlar, prova
6-15 Haziran  Hafta 5: Final polish, prova, repo cleanup
─────────────────────────────────────────────────────────────────────
15 Haziran 2026: FİNAL SEMİNER (40 dakika)
```

---

## 7. Beklenen Çıktılar

| Çıktı | Format | Tahmini Tarih |
|-------|--------|---------------|
| v7-fixed metrik dosyası | JSON (`deeplabv3plus_v7_fixed_metrics.json`) | 10 Mayıs |
| SOTA literatür tablosu | Markdown (`docs/literature_table.md`) | 12 Mayıs |
| Limitations bölümü | Markdown akademik metin | 13 Mayıs |
| Class 4 error analysis | Notebook hücreleri + PNG'ler | 15 Mayıs |
| Multi-seed sonuçları | JSON + tablo | 22 Mayıs |
| Final sunum slaytları | PDF / PowerPoint | 5 Haziran |
| Tez güncellemesi | LaTeX (`docs/tez/`) | 10 Haziran |

---

## 8. Referanslar

1. Alaudah, Y., Michalowicz, P., Alfarraj, M., & AlRegib, G. (2019). A Machine Learning Benchmark for Facies Classification. *Interpretation*, 7(3), SE175–SE187.

2. Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV 2018*.

3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML 2019*.

4. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV 2017*.

5. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV 2016*. (Dice loss kaynağı)

6. Liu, M., Niu, J., et al. (2020). Seismic facies classification using supervised CNNs and semi-supervised GANs. *Geophysics*, 85(4), O47–O58. (F3'te section-based 2D > patch-based 3D bulgusu)

7. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR 2018*.

8. Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR 2017*.

(SOTA karşılaştırma için ek referanslar `docs/literature_table.md`'de derlenmektedir.)
