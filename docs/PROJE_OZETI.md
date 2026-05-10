# Proje Özeti — Sismik Fasiyes Segmentasyonu (Detaylı Devir Belgesi)

> Bu belge, projeye dışarıdan bakan bir yapay zeka modelinin (Claude, GPT, Gemini, vb.) **hiçbir ek dosya açmadan** projenin tüm bağlamını, bugüne kadar denenen şeyleri, bilinen sorunları ve sıradaki adımları anlayabilmesi için yazılmıştır. `docs/AI_HANDOFF.md` dosyasının genişletilmiş halidir; arşivdeki eski sürümleri de kapsar.

**Son güncelleme:** 2026-05-09  
**Aktif sürüm:** v7  
**Hedef:** 15 Haziran 2026 final seminer sunumu (40 dakika, Bilgisayarda Görme dersi yüksek lisans)

---

## 1. Amaç ve Bağlam

### 1.1 Akademik bağlam
- Yüksek lisans öğrencisi (Mehmet Kisli)
- Ders: Bilgisayarda Görme — Bahar 2026
- Çıktı: 40 dakikalık seminer sunumu + LaTeX tezi (`docs/tez/`)
- Final tarihi: 15 Haziran 2026
- Ara sunum tarihi: ~15 Nisan 2026 (geçti — `docs/sunum/` altında PDF'ler mevcut)

### 1.2 Bilimsel problem
**Sismik fasiyes sınıflandırması:** Yeraltı sismik kesitlerinde her pikseli 6 jeolojik fasiyese (örn. tuz, kumtaşı, şeyl tabakaları) atayan piksel-bazlı semantik segmentasyon görevi. Endüstride hâlâ jeofizikçiler tarafından elle yapılır; otomasyon hem zaman hem yorumcu tutarlılığı kazandırır.

### 1.3 Kullanıcı stratejisi
Kullanıcı **dürüst ve eleştirel** geri bildirim istiyor — methodology hatalarını sunumda gizlemek yerine "limitations" başlığında dürüstçe söylemek istiyor. Övgü değil, kritik istiyor.

---

## 2. Veri Seti — Netherlands F3 Block

**Kaynak:** Alaudah et al. 2019, *Interpretation* 7(3) — "A Machine Learning Benchmark for Facies Classification"  
**İndirme:** Zenodo `10.5281/zenodo.3755060` (~1 GB, CC-BY-SA 3.0)  
**Yerel konum:** [data/](data/) — toplam ~1.1 GB

### 2.1 Volüm yapısı
| Dosya | Shape (inline, xline, depth) | Tip | Kullanım |
|---|---|---|---|
| `data/train/train_seismic.npy` | (401, 701, 255) | float64 | Eğitim + val |
| `data/train/train_labels.npy` | (401, 701, 255) | uint8 | Eğitim + val |
| `data/test_once/test1_seismic.npy` | (200, 701, 255) | float64 | Test1 (inline yönü) |
| `data/test_once/test1_labels.npy` | (200, 701, 255) | uint8 | Test1 GT |
| `data/test_once/test2_seismic.npy` | (601, 200, 255) | float64 | Test2 (xline yönü) |
| `data/test_once/test2_labels.npy` | (601, 200, 255) | uint8 | Test2 GT |

Sismik değerler [-1, 1] aralığına önceden normalize edilmiş. Bir "kesit" seçilen eksene göre 2D bir görüntü: inline kesiti = 701×255, crossline kesiti = 401×255 (train hacminde).

### 2.2 6 fasiyes sınıfı ve dengesizliği
| Sınıf | İsim | Train piksel oranı | Zorluk |
|---|---|---|---|
| 0 | Upper North Sea | %28.09 | Kolay (kalın baskın tabaka) |
| 1 | Lower North Sea | %11.89 | Orta |
| 2 | Rijnland | %48.59 | En baskın — model varsayılan tahmini |
| 3 | Scruff | %6.64 | Zor (ince/karışık bant) |
| 4 | Zechstein | %3.28 | Çok zor (Test2'de morfolojik fark) |
| 5 | Under Zechstein | %1.51 | En zor (azınlık + dağınık) |

Bu dengesizlik baseline'da modeli sınıf 2'yi ezbere tahmin etmeye iter — ana methodology zorluğu.

---

## 3. Sürüm Evrimi (v3 → v7)

`archive/` klasöründe v3 hariç her ara sürümün notebook'u korunmuştur. Sayısal sonuçlar `results/sonuc.txt` ve `results/metrics/*.json`'da.

### 3.1 v3 (baseline — arşivde notebook YOK, yalnızca metrikler tutuldu)
- Plain U-Net (~31M parametre), 256×256, weighted CE + Dice, Adam, ReduceLROnPlateau
- **Combined mIoU: 0.4012** — sınıf 2 ezbere baskın, küçük sınıflar çökmüş
- Dosya: yok (eski Colab notebook'u, kaynak silinmiş)
- `docs/plan/PROJE_PLANI.md` bu sürümün spesifikasyonudur

### 3.2 v5 — `archive/deeplabv3plus_v5.ipynb`
İlk büyük atılım. v4 → v5 değişiklikleri (notebook başlığından):
- Encoder: ResNet-50 → **EfficientNet-B4** (ImageNet pretrained)
- ASPP rates: (6,12,18) → **(12, 24, 36)** — daha geniş receptive field
- Image size: 256 → **320×320**
- **WeightedRandomSampler** (S4 crossline odaklı oversample)
- **Mixup α=0.2**
- **Label smoothing eps=0.1**
- Tüm grafikler İngilizce
- **Combined mIoU: 0.7582** — büyük sıçrama (v3'ten +35.7 puan)

### 3.3 v5.3 / v5_3_fixed — `archive/deeplabv3plus_v5.3.ipynb`, `archive/deeplabv3plus_v5_3_fixed.ipynb`
2.5D'ye geçiş ve methodology rafine etme. v3 → v5 başlıklı iyileştirmeler:
1. **2.5D input:** 3 komşu slice → 3 kanal (RGB-benzeri, ImageNet uyumu)
2. **Inline + crossline birlikte eğitim** (ConcatDataset) — Test2 domain shift cevabı
3. **VerticalFlip kaldırıldı** — derinlik ekseni ters çevrilemez (jeofizik kuralı)
4. WeightedRandomSampler S4/S5 ×10 boost
5. Blok-bazlı val split (random yerine contiguous) — spatial leakage'ı azaltma niyeti
6. Focal Loss γ=2 — azınlığa odaklan
7. AdamW + OneCycleLR
8. Gradient accumulation ×4 (efektif batch=32)
9. TTA (HFlip + polarity inversion)

`v5_3_fixed` kısa süre sonra çıktı; ana fark **CosineAnnealingWarmRestarts** scheduler (OneCycle yerine local minima'dan çıkış için).

### 3.4 v6 — `archive/deeplabv3plus_v6.ipynb`
v5.3 metodoloji + v5'in EfficientNet-B4'ü birleşimi. Kayda değer eklemeler:
- Encoder olarak EfficientNet-B4 sabitlendi
- ASPP (12, 24, 36) sabitlendi
- CosineAnnealingWarmRestarts (T_0=25, eta_min=1e-6)
- Early stopping patience=25
- Loss: 0.5·Focal + 0.5·Dice
- Çözünürlük: 256×256 (henüz 320 değil)
- Batch=8

**Sonuç:**
| Metrik | v6 |
|---|---|
| Test1 mIoU | 0.7513 |
| Test2 mIoU | 0.6999 |
| Combined mIoU | 0.7692 |
| Combined Dice | 0.8624 |
| Combined PA | 0.9298 |

### 3.5 v7.0 (arşiv: `archive/deeplabv3plus_v7.0.ipynb`) → v7 (aktif: `deeplabv3plus_v7.ipynb`)
v6 → v7 değişiklikleri:
| Değişiklik | v6 | v7 |
|---|---|---|
| Çözünürlük | 256×256 | **320×320** |
| Mixup | yok | **α=0.2 (p=0.5)** |
| Label smoothing | yok | **eps=0.1** |
| Loss | 0.5·Focal + 0.5·Dice | **0.4·LS-CE + 0.3·Dice + 0.3·Focal** |
| Batch | 8 | **6** (VRAM kısıtı) |
| Diğer | aynı | aynı |

`v7.0` (arşiv) ilk eğitim koşusu. Aktif `deeplabv3plus_v7.ipynb` aynı mimari, sadece **kozmetik path/isim düzeltmeleri** içeriyor (aşağıda 6.1).

---

## 4. v7 Aktif Konfigürasyon (kapsamlı)

### 4.1 Mimari
- **Model:** `segmentation_models_pytorch.DeepLabV3Plus`
  - Encoder: EfficientNet-B4 (ImageNet pretrained)
  - in_channels=3, classes=6
  - Output stride=16, ASPP rates=(12, 24, 36)
- **Girdi:** 2.5D = 3 komşu slice stack'lenip 3 kanal RGB-benzeri görüntü, 320×320'ye resize edilir
- **Toplam parametre:** ~24M (EfficientNet-B4 backbone)

### 4.2 Eğitim
- 100 epoch, AdamW lr=1e-4, weight_decay=1e-3
- CosineAnnealingWarmRestarts (T_0=25, T_mult=1, eta_min=1e-6)
- Loss: **0.4·LabelSmoothing-CE(eps=0.1) + 0.3·DiceLoss + 0.3·FocalLoss(α=class_freq inverse, γ=2)**
- Mixup α=0.2, p=0.5 per batch
- AMP (mixed precision), gradient accumulation ×4 (batch=6 → efektif 24)
- Early stopping patience=25 (val mIoU plateau)
- Checkpoint resume desteği var (`checkpoints_v7/deeplabv3plus_v7_checkpoint.pth`)

### 4.3 Sampling ve augmentation
- **WeightedRandomSampler** — nadir sınıflar (S4 Zechstein, S5 Under Zech) için ×10 boost
- **ConcatDataset:** train inline + train crossline birlikte
- Albumentations: HorizontalFlip, ShiftScaleRotate, ElasticTransform, GridDistortion, RandomBrightnessContrast, GaussNoise, CoarseDropout, polarity inversion (×-1 multiply)
- **VerticalFlip kullanılmaz** — derinlik fiziği

### 4.4 Test Time Augmentation (TTA)
Test sırasında: orijinal + HFlip + polarity inversion → softmax çıktıları averaging.

### 4.5 Mevcut split kodu (cell-9, BUG İÇERİYOR — bkz. Bölüm 5)
```python
val_inline_idx = np.arange(320, 401)         # son %20 contiguous
train_inline_idx = np.arange(0, 320)
train_xline_idx = np.arange(0, 601)          # tüm crossline'lar
```

---

## 5. Mevcut Sayılar (v7)

### 5.1 Test1 / Test2 / Combined
| Metrik | Test1 (inline) | Test2 (xline) | **Combined** |
|---|---|---|---|
| mIoU | **0.7881** | 0.6986 | **0.7926** |
| mean Dice | 0.8726 | 0.7859 | **0.8775** |
| Pixel Accuracy | 0.9356 | 0.9469 | **0.9413** |
| Mean Class Acc | 0.9093 | 0.7963 | 0.8975 |

### 5.2 Per-class IoU (Combined)
| Sınıf | IoU | Dice | Yorum |
|---|---|---|---|
| S0 Upper NS | 0.957 | 0.978 | Mükemmel |
| S1 Lower NS | 0.827 | 0.905 | İyi |
| S2 Rijnland | 0.948 | 0.973 | Beklenen — baskın sınıf |
| S3 Scruff | 0.607 | 0.756 | Zor — ince bant, çok komşu sınırı |
| S4 Zechstein | 0.792 | 0.884 | **Combined iyi ama Test2'de IoU=0.18 — felaket** |
| S5 Under Zech | 0.625 | 0.769 | Azınlık + dağınık morfoloji |

### 5.3 Eğitim sırasında en iyi val mIoU
**0.6617** — yani test1 (0.7881) ile **+12.6 puan ters paradoks**. Normalde test ≤ val olmalı. Bunun nedenleri Bölüm 6'da.

### 5.4 Sürüm karşılaştırması özeti (`results/sonuc.txt`)
```
Metrik                     |    v3   |    v5   |    v6   |    v7
Test1 mIoU (inline)        | 0.6446  | 0.7577  | 0.7513  | 0.7881
Test2 mIoU (crossline)     | 0.2710  | 0.6585  | 0.6999  | 0.6986
Combined mIoU              | 0.4012  | 0.7582  | 0.7692  | 0.7926
Combined Dice              | 0.5453  | 0.8536  | 0.8624  | 0.8775
Combined PA                | 0.6953  | 0.9286  | 0.9298  | 0.9413
```

---

## 6. Bilinen Kritik Sorunlar

### Sorun 1: Val/test paradoksu (val 66% vs test 79%, +12.6 puan)
Test verisinin val'den **kolay** çıkması. Olası nedenler:
- (a) Val seti contiguous block (inline 320–400) → lokasyon bias
- (b) Test1 ayrı volume ama jeolojik olarak val bloğundan daha "uniform"
- (c) TTA testte var, val'de yok (1–2 puan açıklar — 12 puanı değil)
- (d) Sınıf 4 (Zechstein) val bölgesinde **var ama test bölgesinde dağılımı farklı**

### Sorun 2: Methodology bug — Contiguous block val ✅ DÜZELTİLDİ (Yol A)
~~`val_inline_idx = np.arange(320, 401)` son %20'lik düz blok.~~  
**Düzeltildi:** `val_inline_idx = np.arange(160, 240)` — ortadaki 80 inline. Train iki blok halinde: `[0, 158) ∪ [242, 401)`.

### Sorun 3: Methodology bug — 3D crossline leakage ✅ DÜZELTİLDİ (Yol A)
~~`train_xline_idx = np.arange(0, 601)` tüm crossline'lar — val pikselleri içeriyor.~~  
**Düzeltildi:** `train_inline_mask` boolean array; F3Dataset25D `_get_slice` axis=1 modunda val pikselleri xline image'dan çıkarılıyor. Test2 maskelenmedi (ayrı volume).

### Sorun 4: Methodology bug — 2.5D komşu sızıntısı ✅ DÜZELTİLDİ (Yol A)
~~Val inline 320'nin 3 kanalı = [319, 320, 321]. 319 train'de.~~  
**Düzeltildi:** `BUFFER = 2` — train val sınırına 2 inline yaklaşamaz, 2.5D komşu sızıntısı engellenmiş.

### Sorun 5: Bilimsel rigor eksikleri
- **Tek seed (42)**, error bar yok — variance ölçülmemiş
- **Ablation study yok** — Mixup, TTA, 2.5D, rare-class sampling, label smoothing tek tek katkıları ölçülmemiş
- **SOTA literatür karşılaştırması yok** — Alaudah 2019 ve sonrası ile sayısal karşılaştırma tablosu eksik

### Sorun 6: Class 4 (Zechstein) Test2'de çöküş
Test1 IoU = 0.836, Test2 IoU = 0.178 — **4.7 kat fark**. Crossline yönünde Zechstein morfolojik olarak farklı görünüyor (tuz kıvrımı yönelimi); eğitim inline-baskın temsil görüyor. Domain adaptation problemi.

---

## 7. Karar Verilen Düzeltme Planı (Yol A)

```python
# cell-9: Val'i ortaya kaydır + 2 inline buffer
BUFFER = 2
val_inline_idx = np.arange(160, 240)              # ortadaki 80 inline
train_inline_idx = np.concatenate([
    np.arange(0, 160 - BUFFER),
    np.arange(240 + BUFFER, 401)
])

# cell-11: F3Dataset25D crossline modunda train_inline_range'e crop
class F3Dataset25D(Dataset):
    def __init__(self, ..., train_inline_range=None):
        self.train_inline_range = train_inline_range  # (lo, hi) tuple

    def _get_slice(self, vol, idx):
        if self.axis == 0:                          # inline
            return vol[idx]
        else:                                       # crossline
            if self.train_inline_range is not None:
                lo, hi = self.train_inline_range
                return vol[lo:hi, idx]              # val bölgesi hariç
            return vol[:, idx]
```

**Yol B (5-fold CV) reddedildi:** 15–30 saat ekstra GPU; tek seminer için over-engineering.

**Beklenen etki:** val mIoU 2–4 puan düşer (gerçekçileşir), test1/test2 ±1 puan oynar. Sunumda "öncesi/sonrası" tablosu güçlü demonstration.

**Durum:** ✅ **UYGULANDI** (2026-05-09) — git commit `0955d57`. Notebook hücreleri:
- `cell-8` (md), `cell-9` (split + class weights), `cell-11` (F3Dataset25D + train_inline_mask), `cell-12` (compute_slice_weights mask aware), `cell-19` (md), `cell-20` (CHECKPOINT_PATH/BEST_MODEL_PATH → `_fixed`), `cell-22` (JSON output → `deeplabv3plus_v7_fixed_metrics.json`), `cell-31` (md başlık 4-sütun), `cell-32` (v3/v5/v7-broken/v7-fixed delta tablo)

**Eğitim devam ediyor** (başka bilgisayarda). Eski v7 metrikleri (`deeplabv3plus_v7_metrics.json`) ve eski best model (`checkpoints_v7/deeplabv3plus_v7_best.pth`) korunuyor; yeni sonuçlar `_fixed` ekleriyle ayrı dosyalara yazılacak. Sunumda "öncesi vs sonrası" tablosu için ikisi de gerekli.

---

## 8. 2.5D vs 3D Kararı (web araştırması yapıldı)

**Sonuç: 2.5D korunuyor, 3D'ye geçilmeyecek.**

Bulgular:
- Alaudah 2019 benchmark'ın kendisi 2D
- Liu et al. 2020 (*Geophysics* 10.1190/geo2019-0467.1): F3'te section-based 2D > patch-based 3D
- Dou et al. 2023 (*MDPI Remote Sensing*): 2.5D Transformer U-Net > tam 3D U-Net (sentetik fault datası)
- F3'te saf 3D U-Net'in Alaudah benchmark'ını mIoU bazında geçtiği güncel makale **bulunamadı**

3D'nin pratik dezavantajları:
1. Tek volume eğitim verisi → patch-based zorunlu, global jeolojik bağlam parçalanır
2. Sismik için 3D pretrained encoder yok (Med3D tıbbi domain)
3. VRAM patlaması: 16–24 GB+
4. 3D patch sampling sınıf dengesizliğini daha da kötüleştirir

---

## 9. Future Work Adayları (henüz karar verilmedi)

### Aday 1: Foundation Model Fine-tuning (SFM veya GFM) — ARAŞTIRILDI
**SFM (arXiv 2309.02791) — Sheng et al. 2023-2024:**
- Repo: https://github.com/shenghanlin/SeismicFoundationModel (MIT, 153⭐)
- ⚠️ Ağırlıklar **Çin hosting** (Baidu/USTC) — Türkiye'den problem olabilir
- ⚠️ **1-channel grayscale** bekliyor — 2.5D 3-kanallı pipeline'la uyumsuz
- ⚠️ **smp ile uyumlu değil** — kendi ViT + decoder framework'ü
- ⚠️ Eski stack: PyTorch 1.8.1, CUDA 11.1
- Parihaka facies mIoU 0.798 (DeepLab 0.556'yı geçti)
- Effort: 1–2 gün entegrasyon + 8–12 saat retrain

**ThinkOnward GFM — DAHA İYİ ALTERNATİF:**
- Repo: https://github.com/thinkonward/geophysical-foundation-model
- HF: https://huggingface.co/thinkonward/geophysical-foundation-model (Apache 2.0)
- ✅ Hugging Face hosting — 5 dakikada yüklenir
- ✅ Modern PyTorch, ViT-MAE (trace masking)
- ⚠️ Yine 1-channel grayscale (aynı uyumsuzluk)
- F3 facies için bağımsız sonuç **yok** — özgünlük fırsatı (tezde "F3 + foundation model ilk fine-tune")

### Aday 2: Attention U-Net — Geophysics 2024
- F3+Penobscot SOTA iddiası
- `smp.UnetPlusPlus` veya custom attention gate
- Effort: 2–3 saat kod + 6–12 saat retrain
- Beklenen kazanç: +1–3 mIoU puan

### Aday 3: Architecture Comparison Panel (en kolay, en yüksek sunum değeri)
- `smp` ile Unet, UnetPlusPlus, MAnet, FPN, DeepLabV3+ aynı methodology'de yarış
- Effort: 6–8 saat retrain × 4 mimari ≈ 24–32 saat GPU
- **Sunum değeri: yüksek** — bilimsel ablation tablosu

### Aday 4: Domain Adaptation (AdaSemSeg, EarthAdaptNet)
- Test2 0.699 zayıflığını çözmeye yönelik
- Effort: 4–5 gün — single seminer için over-engineering
- **Tavsiye: Future work slaytında bahset, implement etme**

### Önerilen birleşik plan
Methodology fix (Yol A) + 3–4 mimari ablation tek eğitim döngüsünde ≈ 32 saat GPU. Çıktı:

| Mimari | Test1 | Test2 | Combined |
|---|---|---|---|
| DeepLabV3+ (mevcut, bozuk val) | 0.788 | 0.699 | 0.793 |
| DeepLabV3+ (düzeltilmiş) | ? | ? | ? |
| Unet++ | ? | ? | ? |
| MAnet (attention) | ? | ? | ? |
| FPN | ? | ? | ? |

---

## 10. Bekleyen Kararlar / Sıradaki Adımlar

**Şu anda (eğitim devam ederken):**
1. **SOTA literatür karşılaştırma tablosu derlemek** — sunum için kritik (Alaudah 2019, Liu 2020, Shi 2019, Civitarese 2019, SFM, Attention U-Net 2024, AdaSemSeg 2025 sayıları)
2. **Limitations bölümü** akademik metni (Türkçe paragraflar)
3. **PROJE_PLANI.md** güncellemesi (hâlâ U-Net + 256 + 50 epoch yazıyor, gerçek v7'ye uydur)

**Eğitim biter bitmez:**
4. v7-fixed sayılarını yorumla — "öncesi vs sonrası" tablosu üret
5. Class 4 (Zechstein) Test2 IoU=0.18 felaketi için **error analysis görselleştirme** kodu (inline vs crossline morfoloji karşılaştırma, hata heatmap)

**Eğer vakit kalırsa (Hafta 2-3):**
6. **Multi-seed** (seed 42/43/44 → mean±std) — single seed eleştirisini kapat
7. **Mini ablation** (TTA-off, Mixup-off, single-channel) — 3 hızlı koşu
8. **Architecture ablation** (Unet++, MAnet) — methodology-fix üzerinde
9. **ThinkOnward GFM fine-tune** — yüksek ödül, yüksek risk; SOTA tablosu sonrası karar

**Reddedilen / Future Work'te bırakılan:**
- Tam 3D mimari (literatür desteklemiyor — Bölüm 8)
- 5-fold CV (over-engineering)
- Domain adaptation (4-5 günlük iş, future work slaytında)

---

## 11. v7 Notebook Hücre Haritası (`deeplabv3plus_v7.ipynb`)

33 hücre, kritik olanlar:
- **[1–3] Kurulum:** imports, device, paths (`checkpoints_v7/`, `results/figures/`, `results/metrics/`)
- **[4–5] Veri yükleme:** Zenodo download fallback'li
- **[6–7] EDA:** sınıf dağılımı, örnek kesit görselleştirme
- **[8–9] Train/val split + class weights** — ✅ **Yol A uygulandı:** middle val [160,240) + ±2 buffer + train_inline_mask
- **[10–12] F3Dataset25D + WeightedRandomSampler** — ✅ **Yol A uygulandı:** train_inline_mask axis=1 cropping
- **[13–14] Model:** `smp.DeepLabV3Plus(encoder='efficientnet-b4', ...)`
- **[15–16] Loss/Mixup/Optimizer/Scheduler:** triple loss kombinasyonu
- **[17–18] Metrik fonksiyonları:** confusion matrix tabanlı IoU, Dice, PA
- **[19–20] Training loop:** 100 epoch, grad accum, early stop, checkpoint resume
- **[21–22] TTA evaluation + JSON kaydet**
- **[23–30] Görselleştirmeler:** training curves, segmentation 3-panel, confusion matrix, per-class bars
- **[31–32] Sürüm karşılaştırma** (v3/v5/v7)

---

## 12. Tamamlanan Düzeltmeler

### 12.1 Kozmetik (v6→v7 isim/path tutarsızlıkları)
- `checkpoints_v6` → `checkpoints_v7`
- `results_v6/{figures,metrics}` → `results/{figures,metrics}` (README ile uyumlu)
- Tüm matplotlib başlıklarında "v6" → "v7"
- Tüm PNG dosya adlarında "v6" → "v7"
- cell-32 v7 açıklaması: `"EfficientNet-B4 + 2.5D Multi-View + Focal+Dice + TTA"`
- cell-32 path'leri: `results_v5/metrics/...` → `results/metrics/...`

### 12.2 Methodology Fix Yol A (2026-05-09, commit `0955d57`)
- **cell-8** (md): yeni split açıklaması — 3 bug ve fixleri
- **cell-9** (kod): val=[160,240), ±2 buffer, train_inline_mask boolean array, class weights yeni train indices'ten
- **cell-11** (kod): F3Dataset25D class'a `train_inline_mask` parametresi; axis=1 modunda val pikselleri image'dan çıkarılıyor; sadece train_xline_ds maskelendi
- **cell-12** (kod): compute_slice_weights mask aware (sampling weight gerçek eğitim image'ı ile tutarlı)
- **cell-19** (md): eğitim açıklaması güncellendi
- **cell-20** (kod): `BEST_MODEL_PATH` ve `CHECKPOINT_PATH` → `..._fixed_*.pth` (eski model korunur)
- **cell-22** (kod): JSON çıktı → `deeplabv3plus_v7_fixed_metrics.json`; split bilgisi de JSON'a yazılır
- **cell-31** (md): "v3 / v5 / v7-broken / v7-fixed" 4-sütun başlık
- **cell-32** (kod): 4-sütun karşılaştırma + methodology fix delta tablosu

**Eğitim devam ediyor.** Sonuçlar `results/metrics/deeplabv3plus_v7_fixed_metrics.json`'a yazılacak.

---

## 13. Sunum İskeleti (40 dakika)

```
00:00–03:00   Problem: sismik fasiyes nedir, jeofizikçi neden manuel yorumluyor
03:00–07:00   Veri: F3, 6 sınıf, sınıf dengesizliği grafiği
07:00–12:00   Literatür: Alaudah 2019, U-Net/DeepLabV3+ önceki işler, SOTA tablo
12:00–17:00   Mimari: DeepLabV3+ + EfficientNet-B4 + ASPP, neden bu seçim
17:00–24:00   Metodoloji: 2.5D, augmentation, triple loss, mixup, sampling
24:00–30:00   Sonuçlar: ablation tablosu, per-class IoU, görselleştirmeler
30:00–35:00   Karşılaştırma: SOTA tablo + bizim sayılar
35:00–38:00   Limitations: val split bias, leakage, single seed (DÜRÜST OL)
38:00–40:00   Future work + soru
```

---

## 14. Beklenen Zorlu Sorular ve Hazır Cevaplar

1. **"Val %66, test %79 — bu fark nasıl?"**  
   → Val contiguous blok lokasyon bias + TTA testte var val'de yok. **Yol A uygulandı**: val ortaya kaydırıldı, ±2 buffer, xline cropping. Yeni v7-fixed sayıları "öncesi vs sonrası" tablosunda.

2. **"Crossline'lar val inline bölgesinden geçiyor mu?"**  
   → Eski v7'de evet — kabul ettim, **düzelttim**. Yol A'da `train_inline_mask` ile xline image'lar val pikselleri hariç cropped. v7-fixed bu temiz halinde.

3. **"Neden DeepLabV3+, U-Net değil?"**  
   → ASPP ile multi-scale context — sismik tabakaların farklı kalınlıkları için kritik. Architecture ablation tablosu da koyacağım.

4. **"3D mimari neden değil?"**  
   → Tek volume + 3D pretrain yok + Liu 2020 bulgusu (section-based 2D > patch 3D F3'te).

5. **"Mixup'ın katkısını ölçtün mü?"**  
   → Şu an tam ablation yok. Sınırlı ablation (mixup on/off) yapılabilir.

6. **"Sınıf 4 Test2'de IoU 0.18 — neden?"**  
   → Crossline yönünde Zechstein morfolojik farklı, eğitim inline-baskın. Domain adaptation future work.

7. **"%79 mIoU SOTA'ya göre nerede?"**  
   → ⚠️ Şu an cevap yok. SOTA tablosu derlenmesi gerek.

---

## 15. Klasör Yapısı

```
sismik-proje/
├── deeplabv3plus_v7.ipynb        ← AKTİF eğitim notebook'u
├── view_data.ipynb                ← Veri inceleme (eski)
├── README.md
│
├── archive/                       ← Eski sürümler (KORUNUYOR)
│   ├── deeplabv3plus_v5.ipynb     (ResNet-50, ilk büyük atılım)
│   ├── deeplabv3plus_v5.3.ipynb   (2.5D + multi-view geçişi)
│   ├── deeplabv3plus_v5_3_fixed.ipynb  (CosineWR scheduler eklendi)
│   ├── deeplabv3plus_v6.ipynb     (EfficientNet-B4 sabitlendi)
│   └── deeplabv3plus_v7.0.ipynb   (320×320 + Mixup + LS-CE — ilk koşu)
│
├── data/                          ← Zenodo F3 (~1.1 GB)
│   ├── train/{train_seismic.npy, train_labels.npy}
│   └── test_once/{test1_*, test2_*}.npy
│
├── results/
│   ├── sonuc.txt                  ← Sürüm karşılaştırma özeti
│   └── metrics/
│       ├── deeplabv3plus_v5_metrics.json
│       ├── deeplabv3plus_v6_metrics.json
│       └── deeplabv3plus_v7_metrics.json   ← şu anki sayılar
│
├── docs/
│   ├── AI_HANDOFF.md              ← Kısa devir notu
│   ├── PROJE_OZETI.md             ← BU DOSYA (ayrıntılı devir)
│   ├── plan/
│   │   ├── PROJE_PLANI.md         ← v3 dönemi plan (eski U-Net tasarımı)
│   │   ├── LiteraturTaramasi_TR.docx
│   │   └── LiteraturTaramasi_TR_v2.docx
│   ├── sunum/
│   │   ├── DeepLabV3-ile-Sismik-Fasiyes-Siniflandirmasi.pdf  (ara sunum)
│   │   ├── deeplabv3plus v7 teknik aciklama.pdf
│   │   ├── deeplabv3plus v7 FULL CODE ACIKLAMA.pdf
│   │   └── 25812610.pdf            (Alaudah 2019 makalesi)
│   └── tez/
│       ├── SeismicFacies_DeepLabV3plus_Kisli_v2.tex
│       ├── sismik_v3.pdf
│       └── sismik_v4.pdf
│
├── assets/whatsapp/               ← Ekran görüntüleri
└── venv/                          ← Python 3.9 sanal ortam
```

---

## 16. Yeni Yapay Zeka Modeline Görev İsteği

Yukarıdaki bağlamla şu konularda yardım isteyebilirsin (öncelik sırasıyla):

**Eğitim devam ederken (paralel):**
1. **SOTA literatür karşılaştırma tablosu** — Alaudah 2019, Liu 2020, Shi 2019, Civitarese 2019, SFM 2023, Attention U-Net 2024, AdaSemSeg 2025; F3 üzerinde rapor edilen mIoU değerleri
2. **Limitations** bölümü için akademik Türkçe paragraflar (val split bias, leakage, single seed, class 4 Test2)
3. **PROJE_PLANI.md** güncellemesi (v3 dönemi U-Net spec → v7 DeepLabV3+ ile değiştir)
4. **Class 4 Test2 error analysis** kodu (eğitim biter bitmez koşmaya hazır)

**Eğitim bitince:**
5. v7-fixed sayılarını yorumla; "öncesi vs sonrası" tablosu üret
6. Multi-seed (42/43/44) için lightweight koşu planı
7. Mini ablation (TTA-off, Mixup-off, single-channel)
8. Architecture ablation (Unet++, MAnet) — methodology fix üzerinde
9. Sunum slaytları (40 dk) — slayt-bazlı detay (her slaytta hangi grafik/cümle)

**Future work (implementasyona girme):**
10. ThinkOnward GFM fine-tune — yüksek ödül, smp uyumsuz, 1-channel
11. Domain adaptation — Test2 zayıflığı için, future work slaytında bahset

İstediğin formatta yardım edebilirsin: doğrudan kod, slide outline, akademik metin, vb.

---

**Son durum (2026-05-09):**
- v7 mimarisi sabit (eski sayılar: combined mIoU 0.7926, val 0.6617 — paradoks)
- Kozmetik v6→v7 path/isim düzeltmeleri tamamlandı
- ✅ **Methodology fix Yol A UYGULANDI** (cells 8, 9, 11, 12, 19, 20, 22, 31, 32 — commit `0955d57`)
- 🚂 **Eğitim başka bilgisayarda devam ediyor** (3060 Ti, ~1.5–2 saat)
- 📊 Yeni sonuçlar `results/metrics/deeplabv3plus_v7_fixed_metrics.json`'a yazılacak
- 🎯 Sıradaki paralel görev: SOTA literatür tablosu derlemesi
