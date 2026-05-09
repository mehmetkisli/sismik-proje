# AI Modeli Devir Bağlamı — Sismik Fasiyes Projesi

> Bu dosyayı tek seferde başka bir AI modeline (ChatGPT, Gemini, Claude, vb.) yapıştırarak çalışmaya kaldığın yerden devam edebilirsin. Hiçbir dış dosyaya bağımlı değildir.

---

## 1. Bağlam ve Hedef

Yüksek lisans öğrencisiyim, Bilgisayarda Görme dersi (Bahar 2026) için sismik fasiyes segmentasyonu projesi yapıyorum. **40 dakikalık seminer sunumu** hazırlığındayım. Tarih: 15 Haziran 2026 final.

Dürüst, eleştirel feedback istiyorum — methodology hatalarını gizlemek yerine sunumda limitations olarak dürüstçe söylemek istiyorum.

## 2. Proje Teknik Özeti

**Veri:** Netherlands F3 Block (Alaudah 2019 benchmark, Zenodo 3755060)
- Train: `(401, 701, 255)` 3D volume — 401 inline × 701 crossline × 255 depth
- Test1: `(200, 701, 255)` — inline yönü
- Test2: `(601, 200, 255)` — crossline yönü
- 6 fasiyes sınıfı: Upper NS, Lower NS, Rijnland, Scruff, Zechstein, Under Zech
- Ciddi sınıf dengesizliği: S2 Rijnland %48.59, S5 Under Zech sadece %1.51

**Aktif Mimari (v7):**
- DeepLabV3+ + EfficientNet-B4 (ImageNet pretrained)
- 2.5D input: 3 komşu slice = 3 kanal (RGB-benzeri)
- 320×320 resize
- ASPP rates: (12, 24, 36)
- Output stride: 16

**Eğitim Kurulumu:**
- 100 epoch, AdamW lr=1e-4, weight_decay=1e-3
- CosineAnnealingWarmRestarts (T_0=25, eta_min=1e-6)
- Loss: 0.4·LS-CE(eps=0.1) + 0.3·Dice + 0.3·Focal(α=class_freq inverse, γ=2)
- Mixup α=0.2 (p=0.5 per batch)
- Label smoothing eps=0.1
- AMP (mixed precision), gradient accumulation = 4 (batch=6 → efektif 24)
- Early stopping patience=25
- WeightedRandomSampler (rare classes S4, S5 için ×10 boost)
- Augmentation: HorizontalFlip, ShiftScaleRotate, ElasticTransform, GridDistortion, BrightnessContrast, GaussNoise, CoarseDropout, polarity inversion (VerticalFlip YOK — derinlik ekseni)
- Train data: inline + crossline birlikte (ConcatDataset)

**Evaluation:** TTA = original + HFlip + polarity inversion, softmax averaging

## 3. Mevcut Sayılar

| Metrik | Değer |
|---|---|
| Test1 mIoU (inline) | **78.81%** |
| Test2 mIoU (crossline) | 69.86% |
| Combined mIoU | **79.26%** |
| Combined Dice | **87.75%** |
| Combined PA | **94.13%** |
| **Best val mIoU eğitim sırasında** | **66.17%** |

**Per-class IoU (Combined):**
- S0 Upper NS: 0.957
- S1 Lower NS: 0.827
- S2 Rijnland: 0.948
- S3 Scruff: 0.607
- S4 Zechstein: 0.792 (ama Test2'de sadece 0.18 — büyük genelleme açığı)
- S5 Under Zech: 0.625 (en zor sınıf)

**Sürüm gelişimi:** v3 mIoU 0.40 → v5 0.76 → v6 0.77 → v7 0.79

## 4. Tespit Edilen Kritik Sorunlar

### Sorun 1: Val/test paradoksu (12 puan fark)
Best val mIoU = 66.17% iken Test1 = 78.81%. Normalde test ≤ val olmalı. Olası nedenler:
- Val seti contiguous block (lokasyon bias)
- Test1 ayrı volume ama jeolojik olarak val'den daha "benzer"
- TTA testte var, val'de yok (1-2 puan açıklar, 12 puanı değil)

### Sorun 2: Methodology bug — Contiguous block val
```python
val_inline_idx = np.arange(320, 401)  # son %20 düz blok
```
Eğer F3'ün son inline bölgesinde jeoloji farklıysa (Zechstein dağılımı gibi), val sistematik bias içerir → "best epoch" seçimi bozuk.

### Sorun 3: Methodology bug — 3D crossline leakage
```python
train_xline_idx = np.arange(0, 601)  # tüm crossline'lar
```
Her crossline image'ı (n_inlines, depth) shape'inde, yani val_inline_idx pixelleri her bir xline eğitim örneğinde mevcut. **Notebook bunu kabul ediyor ama düzeltmemiş.**

### Sorun 4: Methodology bug — 2.5D komşu sızıntısı
Val inline 320'nin 3 kanalı = [319, 320, 321]. 319 train'de, yani %66'sı train kaynaklı.

### Sorun 5: Bilimsel rigor eksiklikleri
- Tek seed (42), error bar yok
- Ablation study yok (Mixup, TTA, 2.5D, rare-class sampling tek tek ölçülmemiş)
- SOTA literatür karşılaştırması yok

### Sorun 6: Class 4 (Zechstein) felaketi Test2'de
Test1 IoU = 0.84 ama Test2 IoU = 0.18 — 4 kat fark. Crossline yönünde Zechstein'in görünümü morfolojik olarak farklı, eğitim inline-baskın temsil görüyor.

## 5. Notebook Kod Yapısı (`deeplabv3plus_v7.ipynb`)

33 hücre, kritik olanlar:
- **cell-3:** imports, device, paths (`checkpoints_v7`, `results/figures`, `results/metrics`)
- **cell-5:** Veri yükleme (Zenodo download)
- **cell-7:** EDA, sınıf dağılımı
- **cell-9:** Train/val split + class weights — **BUG BURADA**
- **cell-11:** F3Dataset25D class — **BUG BURADA (xline cropping yok)**
- **cell-12:** Rare-class WeightedRandomSampler
- **cell-14:** Model: smp.DeepLabV3Plus
- **cell-16:** Loss (FocalLoss + DiceLoss + TripleLoss) + Mixup + optimizer + scheduler
- **cell-18:** compute_metrics, print_metrics
- **cell-20:** Training loop (100 epoch + grad accum + early stop + checkpoint resume)
- **cell-22:** TTA evaluation + JSON kaydet
- **cell-24/26/28/30:** Görselleştirme (training curves, segmentation comparison, confusion matrix, per-class metrics)
- **cell-32:** v3/v5/v7 versiyon karşılaştırma

## 6. Yapılan Düzeltmeler

Notebook'taki v6→v7 isim/path tutarsızlıkları düzeltildi:
- `checkpoints_v6` → `checkpoints_v7`
- `results_v6/{figures,metrics}` → `results/{figures,metrics}` (README ile uyumlu)
- Tüm matplotlib başlıklarında "v6" → "v7"
- Tüm PNG dosya adlarında "v6" → "v7"
- cell-32 v7 açıklaması README ile uyumlu hale getirildi: `"EfficientNet-B4 + 2.5D Multi-View + Focal+Dice + TTA"`
- cell-32 path'leri düzeltildi: `results_v5/metrics/...` → `results/metrics/...`

## 7. Karar Verilen Methodology Fix Planı (Yol A)

```python
# cell-9: Val'i ortaya kaydır + buffer
BUFFER = 2  # 2.5D komşu sızıntısı için
val_inline_idx = np.arange(160, 240)  # ortadaki 80 inline
train_inline_idx = np.concatenate([
    np.arange(0, 160 - BUFFER),
    np.arange(240 + BUFFER, 401)
])
train_xline_idx = np.arange(0, 601)  # aynı kalır

# cell-11: F3Dataset25D crossline modunda crop
class F3Dataset25D(Dataset):
    def __init__(self, ..., train_inline_range=None):
        self.train_inline_range = train_inline_range  # (min, max) tuple

    def _get_slice(self, vol, idx):
        if self.axis == 0:
            return vol[idx]
        else:
            # Crossline image'ı train inline aralığına crop et
            if self.train_inline_range is not None:
                lo, hi = self.train_inline_range
                return vol[lo:hi, idx]
            return vol[:, idx]
```

**Yol B (5-fold CV) reddedildi:** over-engineering, 15-30 saat GPU gerektirir, single seminer için değer üretmez.

**Beklenen etki:** Val mIoU 2-4 puan düşer (gerçekçileşir), test1/test2 ±1 puan oynar. Sunumda "öncesi vs sonrası" tablosu güçlü demonstration olur.

## 8. 2.5D vs 3D Karar (Web Araştırması Yapıldı)

**Sonuç: 3D'ye geçmeyeceğiz, 2.5D korunacak.**

Web araştırması bulguları:
- Alaudah 2019 benchmark'ın kendisi 2D
- Liu et al. 2020 (Geophysics 10.1190/geo2019-0467.1): F3'te section-based 2D > patch-based 3D
- Dou et al. 2023 (MDPI Remote Sensing): 2.5D Transformer U-Net > tam 3D U-Net (sentetik fault)
- F3 üzerinde saf 3D U-Net'in Alaudah benchmark'ını mIoU bazında geçtiği güncel makale **bulunamadı**

3D'nin pratik dezavantajları:
1. Tek volume eğitim verisi → patch-based zorunlu, global jeolojik bağlam parçalanır
2. Sismik için 3D pretrained encoder yok (Med3D tıbbi)
3. VRAM patlaması: 16-24 GB+
4. Sınıf dengesizliği 3D patch sampling'de daha da kötüleşir

## 9. Future Work Adayları (Tartışıldı, Karar Verilmedi)

**Aday 1: Seismic Foundation Model (SFM)** — arXiv 2309.02791
- 192 survey'den 2.3M slice MAE pretrain ViT
- Parihaka facies mIoU 0.798 (DeepLab 0.556'yı geçti)
- Public weights/code mevcudiyeti **teyit edilmedi** — araştırılmalı
- Effort: 1-2 gün entegrasyon + retrain

**Aday 2: Attention U-Net** — Geophysics 2024
- F3+Penobscot SOTA iddiası
- segmentation_models_pytorch'ta `smp.UnetPlusPlus` veya custom
- Effort: 2-3 saat kod + 6-12 saat retrain
- Beklenen kazanç: +1-3 mIoU puan

**Aday 3: Architecture Comparison Panel (en kolay)**
- smp ile Unet, UnetPlusPlus, MAnet, FPN, DeepLabV3+ karşılaştır
- Effort: Her biri ~6-8 saat retrain
- Sunum değeri: Yüksek — bilimsel ablation tablosu

**Aday 4: Domain Adaptation (AdaSemSeg, EarthAdaptNet)**
- Test2 0.699 zayıflığını çözmeye yönelik
- Effort: 4-5 gün — single seminer için over-engineering
- **Tavsiye: Future work slaytında bahset, implement etme**

**Önerilen birleşik plan:** Methodology fix + 3-4 mimari ablation tek eğitim döngüsünde (~32 saat GPU). Sonuç tablosu:

| Mimari | Test1 mIoU | Test2 mIoU | Combined |
|---|---|---|---|
| DeepLabV3+ (mevcut, bozuk val) | 0.788 | 0.699 | 0.793 |
| DeepLabV3+ (düzeltilmiş) | ? | ? | ? |
| Unet++ | ? | ? | ? |
| MAnet (attention) | ? | ? | ? |
| FPN | ? | ? | ? |

## 10. Bekleyen Kararlar

1. Birleşik plan mı (methodology fix + arch ablation, ~32 saat GPU) yoksa minimal mi (sadece Attention U-Net + methodology fix, ~12 saat)?
2. SFM GitHub'da public weights var mı, araştırılsın mı?

## 11. Sunum Stratejisi (40 dakika)

```
00:00-03:00   Problem: sismik fasiyes nedir, jeofizikçi neden manuel yorumluyor
03:00-07:00   Veri: F3, 6 sınıf, sınıf dengesizliği grafiği
07:00-12:00   Literatür: Alaudah 2019, U-Net/DeepLabV3+ önceki işler, SOTA tablo
12:00-17:00   Mimari: DeepLabV3+ + EfficientNet-B4 + ASPP, neden bu seçim
17:00-24:00   Metodoloji: 2.5D, augmentation, triple loss, mixup, sampling
24:00-30:00   Sonuçlar: ablation tablosu, per-class IoU, görselleştirmeler
30:00-35:00   Karşılaştırma: SOTA tablo + senin sayıların
35:00-38:00   Limitations: val split bias, leakage, single seed (DÜRÜST OL)
38:00-40:00   Future work + soru
```

## 12. Beklenen Zorlu Sorular ve Hazır Cevaplar

1. **"Val %66, test %79 — bu fark nasıl?"**
   → Val contiguous blok lokasyon bias + TTA testte var val'de yok. Düzeltilmiş split ile yeniden koştum (Yol A).

2. **"Crossline'lar val inline bölgesinden geçiyor mu?"**
   → Evet, biliyorum. Bunu düzelttim (xline cropping). Yol A öncesi/sonrası tablo gösterir.

3. **"Neden DeepLabV3+, U-Net değil?"**
   → ASPP ile multi-scale context — sismik tabakaların farklı kalınlıkları için kritik.

4. **"3D mimari neden değil?"**
   → Tek volume + 3D pretrain yok + Liu 2020 bulgusu (section-based 2D > patch 3D F3'te).

5. **"Mixup'ın katkısını ölçtün mü?"**
   → (Ablation yapmadıysa) Henüz tam ablation yok.

6. **"Sınıf 4 Test2'de IoU 0.18 — neden?"**
   → Crossline yönünde Zechstein morfolojik farklı, eğitim inline-baskın. Domain adaptation future work.

7. **"%79 mIoU SOTA'ya göre nerede?"**
   → (Hazırla — şu an cevap yok)

## 13. Görev İsteği (yeni AI modeline)

Yukarıdaki bağlamla aşağıdaki yardımları yapabilirsin:
- Methodology fix kodunu (cell-9 + cell-11) tamamen yeniden yazmak
- Architecture ablation için training loop genelleştirmek
- SOTA literatür karşılaştırma tablosunu derlemek
- Sunum slayt yapısı önerisi
- Limitations bölümü için akademik dil yazımı
- Class 4 Test2 başarısızlığı için error analysis görselleştirme kodu

İstediğim formatta yardım edebilirsin: doğrudan kod, slide outline, akademik metin, vb.

---

**Son durum (Mayıs 2026):** Notebook'taki v6/v7 isim tutarsızlıkları düzeltildi. Sıradaki: methodology fix (Yol A) implementasyonu. Hangi ablation/architecture aday(lar)ını birlikte deneyeceğimize karar verilmedi.
