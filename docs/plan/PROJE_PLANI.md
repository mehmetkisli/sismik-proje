# Derin Öğrenme ile Sismik Fasiyes Sınıflandırması
## Proje Planı — U-Net Tabanlı Segmentasyon

**Ders:** Bilgisayarda Görme (Yüksek Lisans, Bahar 2026)
**Veri Seti:** Netherlands F3 Block — Alaudah et al. (2019)
**Mimari:** U-Net
**Ortam:** Google Colab (T4 GPU) + MacBook M4 (MPS)

---

## 1. Problemin Tanımı

Sismik fasiyes sınıflandırması, yeraltı jeolojik yapılarının sismik dalga yansıma örüntülerine göre kategorilere ayrılması işlemidir. Geleneksel yöntemde jeofizikçiler bu yorumu manuel olarak yapar; bu hem zaman alıcıdır hem de yoruma bağlı farklılıklar içerir.

Bu projede U-Net mimarisi kullanılarak sismik kesitler piksel bazında (6 fasiyes sınıfı) otomatik olarak sınıflandırılacaktır.

**6 fasiyes sınıfı:**

| Sınıf | Piksel Sayısı (Train) | Oran |
|-------|----------------------|------|
| 0     | 20.137.839           | %28.09 |
| 1     | 8.519.666            | %11.89 |
| 2     | 34.831.122           | %48.59 |
| 3     | 4.760.778            | %6.64  |
| 4     | 2.350.150            | %3.28  |
| 5     | 1.081.200            | %1.51  |

**Temel zorluk:** Sınıf 2 tüm piksellerin %48'ini oluştururken, Sınıf 5 yalnızca %1.5'ini oluşturmaktadır. Bu dengesizlik modelin küçük sınıfları görmezden gelmesine yol açar.

---

## 2. Veri Seti

### 2.1 Kaynak

- **Benchmark:** Alaudah, Y. et al. (2019). "A Machine Learning Benchmark for Facies Classification." *Interpretation*, 7(3).
- **İndirme:** Zenodo üzerinden `zenodo.org/record/3755060` adresinden `data.zip` (~1 GB)
- **Lisans:** CC-BY-SA 3.0

### 2.2 Veri Yapısı

İndirilen dosyalar:

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

**Boyut yorumu:** `(inline sayısı, crossline sayısı, derinlik adımı)`

- Train verisi: 401 adet inline kesiti, her biri 701×255 piksel
- Test1: inline yönünde 200 kesit
- Test2: crossline yönünde 601 kesit (farklı boyut — model genellemesi için)

**Sismik değer aralığı:** [-1.0, 1.0] — veri seti önceden normalize edilmiş

### 2.3 Train / Validation / Test Bölmesi

```
Train seti     → 401 inline'dan %80'i = 320 kesit  (model öğrenir)
Validation seti → 401 inline'dan %20'si = 81 kesit  (hiperparametre izlenir)
Test1 seti     → 200 inline                         (final değerlendirme)
Test2 seti     → 601 crossline                      (final değerlendirme)
```

**Kritik kural:** Test setleri eğitim boyunca hiç kullanılmaz. Yalnızca eğitim tamamlandıktan sonra, bir kez çalıştırılır.

Bölme kodu:
```python
indices = np.arange(401)
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
# → train_idx: 320 eleman, val_idx: 81 eleman
```

---

## 3. Teknik Pipeline

### 3.1 Genel Akış

```
Veri İndirme
    ↓
Keşifsel Veri Analizi (EDA)
    ↓
Dataset & DataLoader
    ↓
Model (U-Net)
    ↓
Loss Fonksiyonu + Optimizer
    ↓
Training Loop
    ↓
Validation İzleme + Model Kaydetme
    ↓
Final Test Değerlendirmesi
    ↓
Görselleştirme
```

---

### 3.2 Keşifsel Veri Analizi (EDA)

Yapılacaklar:
- Her sınıfın piksel dağılımını çubuk grafik ile göster
- 3 farklı inline kesitini (baş, orta, son) sismik + etiket yan yana çiz
- Sınıf ağırlıklarını hesapla ve kaydet

---

### 3.3 Dataset ve DataLoader

**Her örnek:**
- Girdi: Bir inline kesiti → shape `(1, 256, 256)` — kanal=1 (gri tonlamalı)
- Etiket: Aynı kesitin piksel etiketleri → shape `(256, 256)` — her piksel 0-5

**Neden 256×256?** Orijinal kesit boyutu 701×255'tir. U-Net 2'nin katları olan boyutları ister. 256×256'ya yeniden ölçekleme `nearest` (etiket) ve `bilinear` (görüntü) interpolasyon ile yapılır.

**DataLoader ayarları:**
```python
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=2)
```

---

### 3.4 U-Net Mimarisi

U-Net 2015 yılında Ronneberger et al. tarafından tıbbi görüntü segmentasyonu için geliştirilmiştir. Skip connection'lar sayesinde hem genel hem yerel özellikler korunur.

**Mimari özet:**

```
Girdi: (batch, 1, 256, 256)
    ↓
Encoder (daralan yol — 4 seviye):
    64 → 128 → 256 → 512 kanal
    Her seviye: DoubleConv + MaxPool
    ↓
Bottleneck: 1024 kanal
    ↓
Decoder (genişleyen yol — 4 seviye):
    Skip connection ile birleştir + DoubleConv
    512 → 256 → 128 → 64 kanal
    ↓
Çıkış konvolüsyonu: 1×1 Conv → 6 kanal
    ↓
Çıktı: (batch, 6, 256, 256)
```

**Toplam parametre:** ~31 milyon

**DoubleConv bloğu:**
```
Conv2d(3×3) → BatchNorm2d → ReLU → Conv2d(3×3) → BatchNorm2d → ReLU
```

---

### 3.5 Loss Fonksiyonu

Sınıf dengesizliği nedeniyle iki loss kombinasyonu kullanılır:

**1. Weighted Cross-Entropy Loss**
Az görülen sınıflara yüksek ceza uygular.

```
Sınıf ağırlıkları (frekans tersine orantılı):
  Sınıf 0: 0.1697    Sınıf 1: 0.4010
  Sınıf 2: 0.0981    Sınıf 3: 0.7176
  Sınıf 4: 1.4537    Sınıf 5: 3.1599
```

**2. Dice Loss**
Örtüşme oranını doğrudan optimize eder; dengesizliğe daha az duyarlıdır.

**Toplam Loss:**
```
Loss = CrossEntropyLoss(weighted) + DiceLoss
```

---

### 3.6 Optimizer ve Scheduler

```python
optimizer = Adam(lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(mode='min', patience=5, factor=0.5)
# Val loss 5 epoch boyunca iyileşmezse lr'yi yarıya indir
```

**Başlangıç learning rate:** 1e-3
**Minimum learning rate:** ~3e-5 (3 indirimden sonra)

---

### 3.7 Training Loop

```
Her epoch için:
│
├── TRAIN aşaması:
│   ├── Her batch: forward → loss → backward → optimizer step
│   └── Epoch train loss = batch loss'ların ortalaması
│
├── VALIDATION aşaması:
│   ├── model.eval() + torch.no_grad()
│   ├── Her batch: forward → val loss
│   └── Her tahmin için IoU ve Dice hesapla
│
├── Metrikleri kaydet (history dict)
├── Scheduler'a val loss ver
└── Eğer val loss en iyisiyse → best_unet.pth kaydet
```

**Toplam epoch:** 50
**Early stopping:** 15 epoch boyunca val mIoU iyileşmezse dur

---

### 3.8 Değerlendirme Metrikleri

Her sınıf için ayrı, sonra ortalama alınır:

| Metrik | Formül | Açıklama |
|--------|--------|----------|
| IoU (Jaccard) | TP / (TP + FP + FN) | Ana segmentasyon metriği |
| mIoU | ortalama(tüm sınıfların IoU'su) | Genel başarı |
| Dice / F1 | 2·TP / (2·TP + FP + FN) | IoU'ya benzer, daha hassas |
| mDice | ortalama(tüm sınıfların Dice'ı) | Genel başarı |
| Pixel Accuracy | doğru piksel / toplam piksel | Basit ama yanıltıcı olabilir |

**Not:** Sınıf 2 çok baskın olduğu için pixel accuracy yüksek çıksa da model başarısız olabilir. mIoU asıl kriter olarak kullanılır.

**Test protokolü:**
1. Eğitim bitince `best_unet.pth` yükle
2. Test1 üzerinde değerlendir (inline yönü)
3. Test2 üzerinde değerlendir (crossline yönü — genelleme testi)
4. Alaudah 2019'daki tablo ile karşılaştır

---

### 3.9 Görselleştirme

**1. Eğitim eğrileri:**
- Train loss / Val loss (epoch bazında)
- Val mIoU (epoch bazında)

**2. Segmentasyon görselleştirmesi:**
Her test inline'ı için 3 sütunlu grafik:
```
[Ham Sismik] | [Ground Truth] | [Tahmin]
```
Renk kodu: Her fasiyes sınıfına sabit renk

**3. Confusion matrix:**
Gerçek sınıf vs tahmin sınıfı — hangi fasiyesler karıştırılıyor

**4. Per-class IoU tablosu:**
Her sınıf için ayrı sonuç

---

## 4. Mevcut Kod Durumu

```
fasiyes.ipynb (Colab)
├── ✅ Veri indirme (Zenodo)
├── ✅ EDA ve görselleştirme
├── ✅ Dataset & DataLoader (256×256 resize)
├── ✅ U-Net mimarisi
├── ✅ Weighted CE + Dice Loss
├── ✅ Adam + ReduceLROnPlateau
├── ⚠️  Training loop yazıldı — henüz çalıştırılmadı
├── ✅ IoU ve Dice metrik fonksiyonları
└── ✅ Model checkpoint kaydetme (best_unet.pth)
```

---

## 5. Yapılacaklar Listesi

### Acil (Bu Hafta)
- [ ] Colab'da training'i başlat, 50 epoch çalıştır
- [ ] Training eğrilerini izle — loss düşüyor mu?
- [ ] Val mIoU logunu kaydet

### Ara Sunum Öncesi (15 Nisan'a kadar)
- [ ] Training tamamla, best_unet.pth kaydet
- [ ] Test1 ve Test2 üzerinde final değerlendirme
- [ ] Per-class IoU tablosu oluştur
- [ ] Segmentasyon görselleştirmeleri hazırla (10+ örnek)
- [ ] Confusion matrix çiz
- [ ] Eğitim eğrisi grafiği

### Ara Sunum İçeriği (~15 Nisan)
- [ ] Problem tanımı slaytı (fasiyes nedir, neden önemli)
- [ ] Dataset slaytı (veri yapısı + sınıf dağılım grafiği)
- [ ] Mimari slaytı (U-Net şeması)
- [ ] Sonuçlar slaytı (metrik tablosu + örnek görseller)

### Final Sunum (~15 Haziran)
- [ ] Yazılı proje raporu
- [ ] Nihai sonuçlar ve yorumlama
- [ ] Alaudah 2019 ile karşılaştırma tablosu

---

## 6. Zaman Çizelgesi

```
24–30 Mart     Colab training başlat, sonuçları izle
31 Mart–6 Nisan   Training tamamla, evaluation kodu çalıştır
7–13 Nisan     Görselleştirme, sunum hazırlığı
──────────────────────────────────────── 15 Nisan: ARA SUNUM
16 Nisan–Mayıs Sonuçları derinleştir, rapor yaz
──────────────────────────────────────── 15 Haziran: FİNAL SUNUM
```

---

## 7. Beklenen Çıktılar

| Çıktı | Format | Tarih |
|-------|--------|-------|
| Eğitilmiş model | `best_unet.pth` | 10 Nisan |
| Metrik tablosu | CSV + slayt | 13 Nisan |
| Segmentasyon görselleri | PNG | 13 Nisan |
| Ara sunum | PowerPoint / PDF | 15 Nisan |
| Proje raporu | PDF (10–15 sayfa) | Haziran |

---

## 8. Referanslar

1. Alaudah, Y., Michałowicz, P., Alfarraj, M., & AlRegib, G. (2019). A Machine Learning Benchmark for Facies Classification. *Interpretation*, 7(3), SE175–SE187.

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.

3. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV 2016*. (Dice loss kaynağı)
