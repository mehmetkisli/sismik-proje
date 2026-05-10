# AI Modeli Devir Bağlamı — Sismik Fasiyes Projesi

> Bu dosyayı tek seferde başka bir AI modeline (ChatGPT, Gemini, Claude, vb.) yapıştırarak çalışmaya kaldığın yerden devam edebilirsin. Hiçbir dış dosyaya bağımlı değildir.

---

## 1. Bağlam ve Hedef

Yüksek lisans öğrencisiyim, Bilgisayarda Görme dersi (Bahar 2026) için sismik fasiyes segmentasyonu projesi yapıyorum. **40 dakikalık seminer sunumu** hazırlığındayım. Tarih: 15 Haziran 2026.

Dürüst, eleştirel feedback istiyorum — methodology hatalarını gizlemek yerine sunumda limitations olarak dürüstçe söylemek istiyorum. **"SOTA"** veya **"Foundation model seviyesinde"** gibi iddialardan uzak duruyorum; bu **F3 üzerinde dürüst, methodology hataları düzeltilmiş bir DeepLabV3+ baseline** sunumu.

## 2. Proje Teknik Özeti

**Veri:** Netherlands F3 Block (Alaudah 2019 benchmark, Zenodo 3755060)
- Train: `(401, 701, 255)` 3D volume — 401 inline × 701 crossline × 255 depth
- Test1: `(200, 701, 255)` — inline yönü
- Test2: `(601, 200, 255)` — crossline yönü
- 6 fasiyes sınıfı: Upper NS, Lower NS, Rijnland, Scruff, Zechstein, Under Zech
- Ciddi sınıf dengesizliği: S2 Rijnland %48.59, S5 Under Zech sadece %1.51

**Aktif Mimari (v7-fixed):**
- DeepLabV3+ + EfficientNet-B4 (ImageNet pretrained)
- 2.5D input: 3 komşu slice = 3 kanal (RGB-benzeri)
- 320×320 resize
- ASPP rates: (12, 24, 36)
- Output stride: 16
- Methodology fix: orta blok val [160, 240) + ±2 inline buffer + crossline image cropping

**Eğitim Kurulumu:**
- 100 epoch (84 epoch'ta early stop), AdamW lr=1e-4, weight_decay=1e-3
- CosineAnnealingWarmRestarts (T_0=25, eta_min=1e-6)
- Loss: 0.4·LS-CE(eps=0.1) + 0.3·Dice + 0.3·Focal(α=class_freq inverse, γ=2)
- Mixup α=0.2 (p=0.5 per batch)
- AMP (mixed precision), gradient accumulation = 4 (batch=6 → efektif 24)
- Early stopping patience=25
- WeightedRandomSampler (rare classes S4, S5 için ×10 boost)
- Augmentation: HFlip, ShiftScaleRotate, Elastic, GridDistortion, BrightnessContrast, GaussNoise, CoarseDropout, polarity inversion (VerticalFlip YOK — derinlik ekseni)
- Train data: inline + crossline birlikte (ConcatDataset)

**Evaluation:** TTA = original + HFlip + polarity inversion, softmax averaging. **Sayılar 320×320 resized space'te hesaplanıyor** — original-resolution evaluator henüz yok.

## 3. Mevcut Sayılar (v7-fixed)

| Metrik | Test1 (inline) | Test2 (crossline) | **Combined** |
|---|---:|---:|---:|
| mIoU | 76.25% | 66.81% | **76.68%** |
| Mean Dice | 85.60% | 75.70% | **86.03%** |
| Pixel Accuracy | 92.28% | 94.10% | **93.19%** |
| Mean Class Acc | 87.43% | 74.93% | **86.15%** |
| FwIoU | 86.34% | 89.63% | **87.84%** |

**Per-class IoU (Combined):**
- S0 Upper NS: 0.937, S1 Lower NS: 0.801, S2 Rijnland: 0.936
- S3 Scruff: 0.601, S4 Zechstein: 0.755, S5 Under Zech: 0.571

**v7-broken (eski, methodology bug'lı sürüm):**
- Combined mIoU 0.7926, FwIoU TBD, val mIoU sadece 0.66 (val/test ters paradoks — leakage göstergesi)
- Bu sayılar contiguous-block val + 3D crossline leakage + 2.5D komşu sızıntısından beslenen yapay yüksek değerlerdi

**Sürüm gelişimi:** v3 mIoU 0.40 → v5 0.76 → v6 0.77 → v7-broken 0.79 (yapay) → v7-fixed 0.77 (gerçek)

## 4. Tespit Edilen Kritik Sorunlar

### Sorun 1: Val/test paradoksu ✅ ÇÖZÜLDÜ
Eski v7'de best val mIoU 0.66 < test1 0.79 (ters paradoks, +12 puan). Methodology fix sonrası best val 0.84 ≥ test 0.76 (normal pattern).

### Sorun 2-4: Methodology bug'ları ✅ ÇÖZÜLDÜ (Yol A)
- ~~Contiguous block val~~ → val [160, 240) ortada
- ~~3D crossline leakage~~ → train_inline_mask ile xline image cropping
- ~~2.5D komşu sızıntısı~~ → ±2 inline buffer

### Sorun 5: Bilimsel rigor eksiklikleri (HÂLÂ AÇIK)
- **Tek seed (42), error bar yok** — varyans ölçülmedi
- **Ablation eksik** — sadece TTA on/off (+0.5 puan); Mixup, 2.5D, xline, rare sampler için ablation yok
- **Original-resolution evaluator yok** — Alaudah ile birebir kıyas için gerekli

### Sorun 6: Class 4 (Zechstein) Test2 felaketi (HÂLÂ AÇIK)
Test1 IoU 0.799, Test2 IoU **0.118** — model %58.8 oranında Under Zechstein ile karıştırıyor. Tuz tabakasının yön bağımlı (anisotropic) morfolojisi; eğitim inline-baskın temsil. **Çözüm methodology değil, domain adaptation gerekiyor.**

### Sorun 7: Block-mask concat fizik komşuluğu (TEKNİK)
v7-fixed'de iki train inline bloğu `[0, 158) ∪ [242, 401)` tek boolean mask ile birleştiriliyor — fiziksel olarak komşu olmayan inlines görüntüde komşu sayılıyor. Resize ve buffer yumuşatsa da temiz çözüm: iki ayrı xline dataset (ConcatDataset).

### Sorun 8: Evaluator resize'a bağlı (TEKNİK)
Sayılar 320×320 resized space'te. Alaudah orijinal çözünürlükte (701×255) evaluator çalıştırır. **Doğrudan kıyas için orijinal-çözünürlük evaluator gerekli** — bekleyen iş.

## 5. Notebook Kod Yapısı (`deeplabv3plus_v7.ipynb`)

33 hücre, kritik olanlar:
- **cell-3:** imports, device, paths (`checkpoints_v7`, `results/figures`, `results/metrics`)
- **cell-5:** Veri yükleme (Zenodo download)
- **cell-7:** EDA, sınıf dağılımı
- **cell-9:** ✅ Methodology-fixed split + class weights
- **cell-11:** ✅ F3Dataset25D + train_inline_mask cropping
- **cell-12:** ✅ compute_slice_weights mask aware
- **cell-14:** Model: smp.DeepLabV3Plus
- **cell-16:** Loss + Mixup + optimizer + scheduler
- **cell-18:** compute_metrics, print_metrics
- **cell-20:** Training loop (BEST_MODEL_PATH = `_fixed_best.pth`)
- **cell-22:** TTA evaluation + JSON kaydet (`deeplabv3plus_v7_fixed_metrics.json`)
- **cell-24/26/28/30:** Görselleştirmeler
- **cell-32:** v3/v5/v7-broken/v7-fixed karşılaştırma + delta tablosu

## 6. Yapılan Düzeltmeler

### v6→v7 Kozmetik (commit önceden)
- `checkpoints_v6` → `checkpoints_v7`
- `results_v6/{figures,metrics}` → `results/{figures,metrics}`
- Tüm matplotlib başlıklarında ve PNG dosya adlarında "v6" → "v7"

### Methodology fix Yol A (commit `0955d57`)
- cell-8/9 (md+kod): orta blok val + buffer + train_inline_mask
- cell-11 (kod): F3Dataset25D `train_inline_mask` parametresi, axis=1 cropping
- cell-12 (kod): compute_slice_weights mask aware
- cell-19/20 (md+kod): checkpoint paths `_fixed_*.pth` (eski model korunur)
- cell-22 (kod): JSON output `deeplabv3plus_v7_fixed_metrics.json`
- cell-31/32 (md+kod): 4-sütun karşılaştırma + delta tablosu

## 7. 2.5D vs 3D Karar (Web Araştırması Yapıldı)

**Sonuç: 3D'ye geçmeyeceğiz, 2.5D korunacak.**

Bulgular:
- Alaudah 2019 benchmark'ın kendisi 2D
- Liu et al. 2020 (Geophysics 10.1190/geo2019-0467.1): F3'te section-based 2D > patch-based 3D
- F3 üzerinde saf 3D U-Net'in Alaudah benchmark'ını mIoU bazında geçtiği güncel makale **bulunamadı**

3D'nin pratik dezavantajları:
1. Tek volume eğitim verisi → patch-based zorunlu, global jeolojik bağlam parçalanır
2. Sismik için 3D pretrained encoder yok
3. VRAM patlaması: 16-24 GB+ (bizde 8 GB var)
4. Sınıf dengesizliği 3D patch sampling'de daha da kötüleşir

## 8. Foundation Model Bağlamı (önemli)

Literatür 2025-2026'da büyük ölçüde **sismik-spesifik foundation model**'lara döndü:

- **SFM (arXiv 2309.02791):** 192 sismik survey, 2.3M slice ile MAE pretrain ViT. Parihaka facies'te DeepLab 0.55 → SFM 0.80 sıçrama.
- **ThinkOnward GFM:** 450 sentetik Synthoseis volume ile ViT-MAE + trace masking. Hugging Face'te public.
- **NCS-Model:** "Doğal görüntü pretraining sismikte güvenilir transfer vermez" diyor.

**Bizim ImageNet EfficientNet-B4 pretrain'imiz** modern literatür perspektifinden **baseline seviyesinde**. SOTA değil. Sunumda dürüst söylenmeli. Foundation model fine-tune future work — implementasyon henüz yok.

## 9. Karşılaştırma — Literatür ile Durum

**Doğrudan kıyaslanabilir tek satır:** Alaudah 2019 best baseline (section + aug + skip):
- PA 0.905, MCA 0.817, FwIoU 0.832 (mIoU rapor edilmemiş)

**Bizim v7-fixed (320×320 resized eval):**
- PA 0.932, MCA 0.861, FwIoU 0.878

Yüzeyde "üç metrikte de geçer" gibi gözüküyor (PA +2.7, MCA +4.5, FwIoU +4.6 puan). **AMA dikkat:**
- Bizim sayılarımız 320×320 resized space'te
- Alaudah'ın evaluator'ü orijinal çözünürlükte (701×255)
- Doğrudan kıyas için **aynı evaluator protokolünü çalıştırmak gerekir** — henüz yapılmadı
- Şu anki iddia: "**Alaudah baseline'a yakın değerler**" diyebilirim, "**geçtim**" demek için orijinal-resolution evaluator lazım

**Yüksek mIoU rapor eden çalışmalar (Wiley 2022 0.94, CONSS 2023 0.95) doğrudan kıyaslanamaz:**
- Random patch split (data leakage)
- Farklı sınıf sayısı (7 vs 6)
- Farklı metrik tanımı

**SFM/GFM bağımsız F3 facies sayısı paylaşmamış** — özgünlük fırsatı ama implementasyon riski yüksek.

## 10. Bekleyen Sıradaki Adımlar

**Öncelik (eleştirel feedback'e göre güncellendi):**
1. **Original-resolution evaluator yaz** — Alaudah ile birebir kıyas için
2. **3 ablation çalıştır:** no-Mixup, 1-channel (2.5D yerine center), inline-only (xline yok)
3. **3 seed (42/43/44)** mean±std raporu
4. **Block-mask concat düzelt** (iki ayrı xline dataset)
5. Class 4 Test2 odaklı: crossline-orientation aug + instantaneous attributes

**Reddedilen / Future Work:**
- Tam 3D mimari (literatür desteklemiyor)
- 5-fold CV (over-engineering)
- Foundation model fine-tune (yüksek effort, sunum sonrası)
- Domain adaptation (4-5 günlük iş)

## 11. Sunum Stratejisi (40 dakika)

```
00:00-03:00   Problem: sismik fasiyes nedir, neden manuel yorumlama
03:00-07:00   Veri: F3, 6 sınıf, sınıf dengesizliği grafiği
07:00-12:00   Literatür: Alaudah 2019, foundation model trendi, baseline tanımı
12:00-17:00   Mimari: DeepLabV3+ + EfficientNet-B4 + ASPP, 2.5D
17:00-24:00   Metodoloji: triple loss, mixup, sampling, methodology fix anlatısı
24:00-30:00   Sonuçlar: per-class IoU, "öncesi vs sonrası" tablosu
30:00-35:00   Karşılaştırma: Alaudah baseline'a yakın değerler, evaluator caveat'i
35:00-38:00   Limitations: val split, leakage, single seed, evaluator resolution, Class 4 Test2
38:00-40:00   Future work + soru
```

**Ana mesaj (savunulabilir iddia):**
> "F3 Hollanda benchmark'ı üzerinde, methodology hatalarını analiz edip düzelten, dürüstçe değerlendirilen bir DeepLabV3+ baseline. Alaudah 2019 baseline'ına yakın değerler. Azınlık sınıf ve yön-bağımlı genelleme hâlâ açık — domain adaptation gerekiyor."

**Demeyeceğim:**
- ❌ "SOTA"
- ❌ "Foundation model seviyesinde"
- ❌ "Alaudah'ı geçtik" (evaluator protokolü uyumlu değil henüz)
- ❌ "Genelleyen sismik yorumlama sistemi"

## 12. Beklenen Zorlu Sorular ve Hazır Cevaplar

1. **"Val %66, test %79 nasıl?"** — Eski v7'de leakage vardı. Methodology fix Yol A ile düzelttim. Yeni: best val 0.84 ≥ test 0.76, normal pattern. Eski mIoU 0.79 yapay olarak yüksekti.

2. **"Alaudah'ı geçtin mi?"** — PA/MCA/FwIoU sayılarımız Alaudah baseline'ından yüksek görünüyor (0.932/0.861/0.878 vs 0.905/0.817/0.832). Ancak biz 320×320 resized space'te ölçüyoruz, Alaudah orijinal çözünürlükte. **Aynı evaluator protokolüyle birebir kıyas henüz yapılmadı** — bunu yaptıktan sonra net iddia üretebilirim.

3. **"3D mimari neden değil?"** — Tek volume + 3D pretrain yok + Liu 2020 bulgusu (section-based 2D > patch 3D F3'te).

4. **"Mixup'ın katkısını ölçtün mü?"** — Şu an sadece TTA on/off var. Mixup, 2.5D, xline ablation planlandı.

5. **"Sınıf 4 Test2'de IoU 0.12 — neden?"** — Crossline yönünde Zechstein anisotropic morfolojide, eğitim inline-baskın temsil. Confusion: %58.8 Under Zech ile karıştırılıyor. Domain adaptation problemi (future work).

6. **"Foundation model neden denenmedi?"** — Literatürde SFM/GFM trendinde olduğunu biliyorum. Smp uyumsuzluğu + 1-channel grayscale kısıtı + kurulum zamanı nedeniyle bu seminer kapsamı dışında bıraktım. Future work kategorisinde.

7. **"%77 mIoU SOTA'ya göre nerede?"** — Mevcut çalışma SOTA iddiasında değil. Modern literatür foundation model trendinde; bizim ImageNet pretrain EfficientNet-B4 baseline seviyesinde. F3 baseline'ı (Alaudah 2019) ile yakın değerlerde.

## 13. Görev İsteği (yeni AI modeline)

Yukarıdaki bağlamla şunlarda yardım isteyebilirsin:
- Original-resolution evaluator scripti yazımı
- Üç ablation için notebook config-driven hâle getirme
- 3-seed eğitim manifesti
- Block-mask concat düzeltmesi (iki ayrı xline dataset)
- Class 4 Test2 odaklı (crossline aug, instantaneous attributes)
- Sunum slaytları + akademik dürüst dil
- Tezin uzun versiyonu için ek ablation matrisi planlaması

İstediğin formatta yardım edebilirsin: doğrudan kod, slide outline, akademik metin, vb. Türkçe konuşalım.

---

**Son durum (2026-05-10):**
- v7-fixed eğitim tamamlandı, sayılar yukarıda
- Eleştirel external feedback alındı: doc tutarsızlığı + evaluator caveat + ablation eksikliği
- Sunum dili "SOTA" / "geçtik" iddialarından arındırılıyor
- Sıradaki: original-resolution evaluator + 3 ablation + multi-seed
