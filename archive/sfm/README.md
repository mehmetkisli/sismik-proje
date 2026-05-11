# SFM (Seismic Foundation Model) Deneyleri — Arşiv

> **Bu klasör arşivdir.** SFM ile yapılan 3 deneyin tüm kod, sonuç ve veri dosyaları burada toplanmıştır. **Ana model SFM değil, v9 (`deeplabv3plus_v9.ipynb`) kullanılmaktadır.** Bu README, nedenlerini ve elde edilen tüm sayıları detaylı belgeler.

**Tarih:** 2026-05-10 / 2026-05-11
**Pretrained kaynak:** [SFM Base-512.pth (Sheng et al. 2024, *Geophysics*)](https://github.com/shenghanlin/SeismicFoundationModel) — USTC linkinden manuel indirildi

---

## 1. Neden SFM denendi?

Literatür araştırmasında (ana proje `docs/literature_table.md`) net bir bulgu çıktı:

- **F3 Alaudah split'inde hiçbir sismik foundation model (SFM, GFM, GEM) doğrulanmış sayı raporlamamış**
- Bu, akademik literatürde gerçek bir boşluk — 2026 itibarıyla doldurulabilir bir niş
- SFM Parihaka veri setinde DeepLab 0.556 → SFM 0.798 sıçraması göstermiş (+24p)
- Hipotez: F3'te de benzer sıçrama olabilir, özellikle Class 4 (Zechstein) Test2 sorununda

**Beklenen kazanç:** Combined mIoU v9'un 0.7769'undan **+5-15 puan** + tezde "F3 + sismik foundation model ilk fine-tune" özgün katkı.

---

## 2. Yapılan 3 Deney

### 2.1 Fizibilite testi (`feasibility_test.py`)

**Amaç:** SFM weights gerçekten yüklenebiliyor mu, forward pass çalışıyor mu, kısa eğitim sağlıklı çıktı veriyor mu?

**Konfigürasyon:**
- SFM Base-512 pretrained yüklendi (pos_embed 512→384 interpolation)
- Encoder **frozen** (sadece channel_adapter + decoder eğitilir, ~4.2M / 89.9M trainable)
- 10 epoch, basit CE + Dice loss
- Sade TTA yok
- ~1 saat eğitim

**Sonuç:** Combined mIoU **0.6929** (v9: 0.7769, **-8.4p**).
- Test1 mIoU: 0.7070
- Test2 mIoU: 0.6073
- Class 4 Test2: 0.1839

**Yorum:** Frozen encoder modunda v9'un altında, ama val mIoU çok hızlı yükseldi (Ep1 0.70 → Ep9 0.77 — random init olsa 0.50 civarı olurdu). SFM pretrain bilgisi açıkça aktarılıyor → full fine-tune denenmesi anlamlı.

**Sayısal sonuç:** [`metrics/sfm_feasibility_metrics.json`](metrics/sfm_feasibility_metrics.json)

---

### 2.2 SFM v1 (`full_finetune.py`)

**Amaç:** Full encoder unfreeze + layer-wise LR decay + Lovász + xline-aware augmentation.

**Konfigürasyon:**
- Encoder **unfrozen** (89.9M params trainable)
- Layer-wise LR decay: `BASE_LR_ENC = 5e-5`, `LAYER_DECAY = 0.75`
  - **Etkin encoder lr (en derin blok 0):** 5e-5 × 0.75^12 ≈ **1.5e-6** (çok düşük!)
- LR head: 1e-3 (channel_adapter + decoder)
- QuadrupleLoss: 0.30 LS-CE + 0.25 Dice + 0.25 Focal + 0.20 Lovász (FP32 wrap)
- 60 epoch, patience 15 → Ep 36'da early stop, **best Ep 21 val mIoU 0.7896**
- Mixup α=0.2, xline-aware aug
- ~70 dakika eğitim

**Sonuç (sade TTA — HFlip + polarity):**

| Metrik | SFM v1 | v9 | Δ |
|---|---|---|---|
| Combined mIoU | 0.7281 | 0.7769 | −4.88p |
| Test1 mIoU | 0.7209 | 0.7752 | −5.43p |
| Test2 mIoU | 0.6665 | 0.6869 | −2.04p |
| **Class 4 Test2** | **0.2699** | 0.2304 | **+3.95p** 🏆 |
| Class 5 Test2 | 0.5081 | 0.6510 | −14.29p |

**Per-class IoU (Combined):**
| Sınıf | v1 | v9 |
|---|---|---|
| S0 Upper NS | 0.9280 | 0.9433 |
| S1 Lower NS | 0.7909 | 0.7968 |
| S2 Rijnland | 0.9280 | 0.9356 |
| S3 Scruff | 0.5329 | 0.6008 |
| S4 Zechstein | 0.7162 | 0.7838 |
| S5 Under Zech | 0.4727 | 0.6010 |

**Önemli bulgu:** **Class 4 Test2'de v9'u net olarak geçti (+%17 göreli).** Ama trade-off var: Class 5 Test2 -14.29p düştü. Model Class 4 boost'una yöneldikçe Class 5'i ezdi.

**Sürüm evrimi (Class 4 Test2):** v7-fixed 0.118 → v7-c4fix 0.182 → v9 0.230 → **SFM v1 0.270** (v7-fixed'in 2.3 katı).

**Sayısal sonuç:** [`metrics/sfm_finetune_metrics.json`](metrics/sfm_finetune_metrics.json)

---

### 2.3 SFM v2 (`full_finetune_v2.py`)

**Amaç:** v1'in encoder neredeyse-frozen kaldığı tespit edildi (lr 1.5e-6 çok düşük). v2'de encoder daha agresif fine-tune edilecek.

**v1 → v2 değişiklikleri:**
| Parametre | v1 | **v2** |
|---|---|---|
| BASE_LR_ENC | 5e-5 | **2e-4** (4× artış) |
| LAYER_DECAY | 0.75 | **0.9** (daha az agresif decay) |
| **Etkin encoder lr (blok 0)** | 1.5e-6 | **2.8e-5** (18× artış) |
| PATIENCE | 15 | **20** |
| NUM_EPOCHS | 60 | **80** |
| RARE_BOOST | 10 | **15** (Class 4+5 birlikte) |

**Sonuç:**
- 42 epoch'ta early stop, **best Ep 22 val mIoU 0.8233** ← **v9'un 0.8022'sini geçti** ✓

| Metrik | SFM v2 | v9 | Δ |
|---|---|---|---|
| Combined mIoU | 0.7677 | 0.7769 | −0.92p (denk) |
| Test1 mIoU | **0.7823** | 0.7752 | **+0.71p** 🏆 |
| Test2 mIoU | 0.6591 | 0.6869 | −2.78p |
| **Class 4 Test2** | 0.2011 | 0.2304 | **−2.93p** ⚠️ |
| Class 4 Combined | **0.8049** | 0.7838 | +2.11p 🏆 |
| **Best val mIoU** | **0.8233** | 0.8022 | **+2.11p** 🏆 |

**Beklenmedik bulgu (tezde değerli):** Encoder agresif fine-tune Class 4 Test2'de **zarar verdi** (v1 0.27 → v2 0.20). Açıklama: encoder full öğrenince inline-baskın training Class 4 morfolojisini "Test1'e uygun" temsil ettirip Test2 genellemesini bozdu. v1'in frozen-style encoder yaklaşımı bu bias'tan korunmuştu.

**Sayısal sonuç:** [`metrics/sfm_finetune_v2_metrics.json`](metrics/sfm_finetune_v2_metrics.json)

---

## 3. Trade-off Matrisi — Ana Bulgu

Üç model arasında **tek lider yok**; her birinin spesifik avantajı var:

| Metrik | Lider | Değer |
|---|---|---|
| Combined mIoU | **v9** | 0.7769 |
| Test1 mIoU | **SFM v2** | 0.7823 |
| Test2 mIoU | **v9** | 0.6869 |
| **Class 4 Test2** | **SFM v1** | **0.2699** |
| Class 4 Combined | **SFM v2** | 0.8049 |
| Class 5 Test2 | **v9** | 0.6510 |
| Best val mIoU | **SFM v2** | 0.8233 |

---

## 4. Neden Ana Model SFM Değil v9 Kaldı?

Bu klasörün en önemli sorusu. **5 ana neden:**

### 4.1 Combined mIoU'da v9 lider
Sayısal olarak v9 0.7769 > SFM v2 0.7677 > SFM v1 0.7281. SFM v2 v9'a çok yakın (−0.92p) ama yine de altında.

### 4.2 SFM eğitimi çok daha pahalı
- v9: tek notebook, ~1.5 saat eğitim
- SFM v1: ~70 dk eğitim
- SFM v2: ~80 dk eğitim
- Toplam SFM yatırımı: ~3 saat + manuel weights indirme + entegrasyon kod yazımı + Windows pickle fix + 1-channel adapter + pos_embed interpolation + multi-scale TTA crash + sade TTA fallback
- v9 daha **stable**, daha **reprodüktif**, daha **basit** bir pipeline

### 4.3 Class 5 Test2 trade-off (v1) ve Class 4 Test2 düşüşü (v2)
- v1: Class 4'te kazandı (+3.95p) ama Class 5'te -14.29p kaybetti
- v2: Class 4 Combined iyi (+2.11p) ama Class 4 Test2 (en kritik metrik) v9'un altına düştü
- v9 dengelidir — net olarak en az "köşeli" model

### 4.4 Akademik dürüstlük + tezdeki katkı
"F3 + sismik foundation model ilk fine-tune" iddiası **tezin Discussion bölümünde değerli** ama **ana model olarak v9 daha güvenli** çünkü:
- v9 mimari daha tanıdık (DeepLabV3+ + EfficientNet-B4 + 2.5D)
- v9 sayıları daha geniş validation profiline sahip
- v9 multi-scale TTA çalışıyor (SFM'de ViT fixed img_size nedeniyle yok)
- Tezde "v9 ana model, SFM ablation/Discussion" daha güçlü hikaye

### 4.5 SFM teknik kısıtları
- ViT fixed img_size — multi-scale TTA yapılamaz (sade TTA fallback'i kullanıldı)
- 1-channel pretrain — bizim 5-kanal 2.5D'yi compress etmek (`5→1 conv`) bir miktar bilgi kaybı
- PyTorch 1.8 → 2.6 uyumsuzluğu (Linux PosixPath pickle hatası, monkey-patch'le çözüldü)
- Pretrained weights resmi olarak sadece USTC (Çin) + Baidu üzerinden — HuggingFace mirror yok, gated, Türkiye'den manuel indirme zor

---

## 5. Tezde Nasıl Sunulacak?

**Ablation tablosu (önerilen yapı):**

| Yöntem | Combined mIoU | Test1 | Test2 | C4 Test2 | Not |
|---|---|---|---|---|---|
| v7-broken (eski metodoloji) | 0.793 | 0.788 | 0.699 | 0.178 | Leakage'lı — sunulamaz |
| v7-fixed (Yol A) | 0.767 | 0.762 | 0.668 | 0.118 | Dürüst baseline |
| v7-c4fix (Class 4 paket) | 0.778 | 0.790 | 0.667 | 0.182 | +Lovász, +5ch, +xline-aug |
| **v9 (ana model)** | **0.777** | 0.775 | 0.687 | 0.230 | + 384×384, +ms-TTA |
| SFM v1 (frozen encoder) | 0.728 | 0.721 | 0.667 | **0.270** | F3'te SFM ilk fine-tune; Class 4 lideri |
| SFM v2 (full unfrozen) | 0.768 | **0.782** | 0.659 | 0.201 | Best val mIoU 0.823; agresif tuning |

**Sunum/tez argümanı:**

> "Sismik-spesifik foundation model (SFM) F3 Alaudah split'inde literatürdeki ilk doğrulanmış sonucu sunuyoruz. SFM v1 frozen-style fine-tune ile Class 4 (Zechstein) Test2 IoU'sunu 0.230'dan 0.270'e (göreli +%17) yükseltti — sismik domain pretrain bilgisinin anisotropic tuz morfolojisinde değer kattığının kanıtı. Ancak agresif encoder fine-tune (SFM v2) Class 4 Test2'de gerileme yaratırken Test1 ve val'de v9'u geçti — bu trade-off, foundation model'in nasıl uygulanması gerektiğine dair değerli bir tasarım dersi sundu. Ana modelimiz v9 olarak kalmaya devam ediyor; SFM sonuçları ablation tablosunda raporlanıyor."

---

## 6. Yeniden Çalıştırma Talimatı (Reproducibility)

Scriptler arşivdeyken çalıştırıldı; yeniden çalıştırmak için:

```bash
# 1. Pretrained weights yerinde mi kontrol et:
ls archive/sfm/pretrained/SFM-Base-512.pth   # 1020 MB

# 2. Path düzeltmesi gerekli (scriptler PROJECT_DIR'i hesaplarken artık archive/ içinde):
#    Her scriptin başında:
#    PROJECT_DIR = Path(__file__).parent.parent  →  Path(__file__).parent.parent.parent
#    PRETRAINED = ... / "sfm" / ...  →  ... / "archive" / "sfm" / "pretrained" / ...

# 3. Veya direct çalıştır (path düzenleme sonrası):
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe archive/sfm/full_finetune.py
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe archive/sfm/full_finetune_v2.py
```

**Not:** Yerel checkpoint dosyaları `checkpoints_v7/` altındadır (gitignore'lu, repo'da yok):
- `sfm_finetune_best.pth` (v1, 343 MB)
- `sfm_finetune_v2_best.pth` (v2, 343 MB)
- Resume checkpoint'leri (1.1 GB each) — pratik olarak gereksiz, silinebilir

---

## 7. Dosya Yapısı

```
archive/sfm/
├── README.md                    ← bu dosya
├── code/
│   ├── models_segmentation.py   ← SFM resmi VIT_MLAHead + ViT-Base (yerel kopya)
│   └── sfm_wrapper.py           ← 5→1 channel adapter + pos_embed interp + Win pickle fix
├── feasibility_test.py          ← Deney 1: frozen 10 epoch sanity
├── full_finetune.py             ← Deney 2: SFM v1 (encoder lr 5e-5, layer_decay 0.75)
├── full_finetune_v2.py          ← Deney 3: SFM v2 (encoder lr 2e-4, layer_decay 0.9)
├── eval_only.py                 ← v1 multi-scale TTA crash sonrası eval düzeltmesi
├── pretrained/
│   └── SFM-Base-512.pth         ← 1.0 GB, USTC linkinden indirildi (gitignore'lu)
└── metrics/
    ├── sfm_feasibility_metrics.json
    ├── sfm_finetune_metrics.json       (v1 sonuçları)
    └── sfm_finetune_v2_metrics.json    (v2 sonuçları)
```

---

## 8. Referanslar

- **SFM paper:** Sheng et al. 2024, "Seismic Foundation Model (SFM): a next-generation deep-learning model in geophysics", *Geophysics* — [arXiv:2309.02791](https://arxiv.org/abs/2309.02791)
- **SFM GitHub:** [github.com/shenghanlin/SeismicFoundationModel](https://github.com/shenghanlin/SeismicFoundationModel)
- **SFM weights:** USTC link (1.0 GB) — sayfada Baidu mirror'ı da var
- **MAE pretrain temel:** He et al. 2022 — Masked Autoencoders Are Scalable Vision Learners
- **Lovász-Softmax:** Berman et al. 2018, CVPR
- **Bizim ana proje:** [`README.md`](../../README.md), [`docs/PROJE_OZETI.md`](../../docs/PROJE_OZETI.md), [`docs/literature_table.md`](../../docs/literature_table.md)
