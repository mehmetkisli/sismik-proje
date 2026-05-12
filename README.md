# Sismik Fasiyes Sınıflandırması — DeepLabV3+

Bu proje, F3 Hollanda sismik veri kümesi (Alaudah 2019 benchmark) üzerinde DeepLabV3+ (EfficientNet-B4) tabanlı semantik segmentasyon ile fasiyes sınıflandırması yapar.

## Aktif Sürüm: v9

**v9 = methodology-fix + Class 4 paket + 384×384 + multi-scale TTA**
- 5 kanallı 2.5D (±2 komşu)
- QuadrupleLoss: LS-CE + Dice + Focal + Lovász-Softmax
- xline-aware augmentation (Class 4 morfolojisi için agresif elastic+grid)
- 384×384 çözünürlük
- Multi-scale TTA (HFlip + polarity + scale 0.75/1.0/1.25)
- Yol A methodology fix (orta blok val + ±2 inline buffer + crossline cropping)

| Metrik | Test1 (inline) | Test2 (crossline) | **Combined** |
|---|---:|---:|---:|
| mIoU | 77.52% | 68.69% | **77.69%** |
| Mean Dice | 86.52% | 78.43% | **86.75%** |
| Pixel Accuracy | 92.81% | 94.08% | **93.45%** |
| Mean Class Acc | 87.20% | 77.13% | **86.27%** |

**Sürüm evrimi (Combined mIoU):**
| Sürüm | Combined mIoU | Önemli yenilik |
|---|---:|---|
| v3 (baseline U-Net) | 0.4012 | Plain U-Net, ImageNet yok |
| v5 | 0.7582 | EfficientNet-B4 + DeepLabV3+ + 320×320 |
| v6 | 0.7580 | + 2.5D + WeightedSampler |
| v7-broken | 0.7926 | ⚠️ methodology bug'ları (leakage'lı) |
| v7-fixed | 0.7668 | + Yol A methodology fix |
| v7-c4fix | 0.7779 | + 5-channel + Lovász + xline-aware aug |
| v9 (single-seed) | 0.7769 | + 384×384 + multi-scale TTA |
| **v9 multi-seed ensemble** | **0.7910 ± 0.006** | + 3-seed (42+43+44) softmax averaging |

> **v9 Multi-seed Ensemble (aktif final model):** 3 bağımsız seed ile eğitim + softmax averaging → Combined mIoU **0.7910 ± 0.006**, v9 tek-seed'ten **+1.41p**. Detaylar: [`ensemble/README.md`](ensemble/README.md).
>
> **Kritik akademik bulgu:** v9 tek-seed Class 4 Test2 sonucu (0.230) **istatistiksel outlier'mış** — 3-seed gerçek değer 0.183 ± 0.04. Tek-seed bias riskinin net kanıtı, tezde akademik dürüstlük örneği.
>
> **SFM deneyleri (arşiv):** Sismik Foundation Model (Sheng et al. 2024) ile 3 ek deney yapıldı (`archive/sfm/`). Sayısal olarak ana modelimiz v9'u tam geçmedi ama Class 4 Test2'de literatür için değerli bir bulgu ürettik (SFM v1: 0.270, v9'un 0.230'undan +%17 göreli iyileşme). Detaylar + neden ana model olarak seçilmedi: [`archive/sfm/README.md`](archive/sfm/README.md).

> **v7-broken not:** 0.7926 görünüyor ama contiguous-block val + 3D crossline leakage + 2.5D komşu sızıntısından beslenen yapay yüksek değer. Methodology fix sonrası gerçek genelleme performansı ortaya çıktı.
>
> **Önemli kavet:** Yukarıdaki sayılar 320×320 / 384×384 resized space'te hesaplanmıştır. Alaudah benchmark'ı orijinal çözünürlükte (701×255 / 401×255) evaluator çalıştırır; "geçtik" iddiası için orijinal-çözünürlük evaluator gerekir (bekleyen iş).

## Class 4 (Zechstein) Test2 — odak metriği

Test2'de Class 4 (tuz tabakası), morfolojinin yön bağımlılığından dolayı tüm sürümlerin en zayıf noktasıydı. Sürüm bazlı IoU evrimi:

| Sürüm | Class 4 Test2 IoU |
|---|---:|
| v7-fixed | 0.1177 |
| v7-c4fix | 0.1820 (+6.43p, göreli +%55) |
| **v9** | **0.2304** (+11.27p vs v7-fixed, **göreli +%96**) |

Detaylı hata analizi: `results/figures/error_analysis/`

## Bilinen Sınırlılıklar

- **Class 4 (Zechstein) Test2 IoU 0.230** — v7-fixed'in 0.118'inden ~2 katına çıktı, ama hâlâ Test1 (0.82) ile uçurum var. Tuz tabakasının yön bağımlı (anisotropic) morfolojisi; çözüm için domain adaptation veya sismik-spesifik foundation model fine-tune (future work).
- **Tek seed (42)** — error bar yok, varyans ölçülmedi. Multi-seed planlandı.
- **Ablation eksik** — TTA on/off var, ama Lovász, 5-channel, xline-aware aug için ayrı ablation yok.
- **Evaluator resize'a bağlı** — 384×384 metrikler. Original-resolution evaluator henüz yok.
- **AMP NaN riski** — 384×384'te Lovász+AMP fp16 NaN üretti, FP32 wrap ile çözüldü; raporlanması gereken bir nüans.

Detaylı liste: [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

## Klasör Yapısı

```
.
├── deeplabv3plus_v9.ipynb         ← Aktif eğitim notebook'u (v9)
├── view_data.ipynb                ← Veri inceleme
│
├── ensemble/                      ← Multi-seed ensemble (3 seed: 42+43+44)
│   ├── README.md                  ←   detaylı ensemble belgesi
│   ├── train_v9_seed.py           ←   SEED parameter ile v9 eğitimi
│   └── evaluate_ensemble.py       ←   softmax averaging inference
│
├── archive/                       ← Eski sürümler + SFM deneyleri
│   ├── deeplabv3plus_v*.ipynb     ←   v5/v5.3/v6/v7.0/v7/v7-c4fix/v8-exp
│   └── sfm/                       ←   Sismik Foundation Model deneyleri (v1, v2)
│
├── data/                          ← Veri seti (notebook bu yoldan okuyor)
│   ├── train/
│   │   ├── train_seismic.npy      (547 MB)
│   │   └── train_labels.npy
│   └── test_once/
│       ├── test1_seismic.npy
│       ├── test1_labels.npy
│       ├── test2_seismic.npy
│       └── test2_labels.npy
│
├── checkpoints_v7/                ← v7 ailesi + v9 best/checkpoint .pth dosyaları
│
├── results/
│   ├── sonuc.txt                  ← Sürüm karşılaştırma özeti
│   ├── metrics/                   ← v3/v5/v6/v7-*/v9 JSON metrikleri
│   └── figures/                   ← Eğitim/test/error_analysis görselleri
│
├── scripts/
│   └── class4_error_analysis.py   ← Class 4 hata analiz scripti
│
├── docs/
│   ├── PROJE_OZETI.md             ← Detaylı devir belgesi
│   ├── AI_HANDOFF.md              ← Kısa devir notu
│   ├── LIMITATIONS.md             ← Bilimsel sınırlılıklar
│   ├── literature_table.md        ← SOTA karşılaştırma tablosu
│   ├── tez/                       ← LaTeX kaynağı + PDF çıktıları
│   ├── sunum/                     ← Sunum PDF'leri + SUNUM_SCRIPTI.md
│   └── plan/                      ← Proje planı + literatür taraması
│
├── assets/whatsapp/               ← Ekran görüntüleri / fotoğraflar
│
└── .venv/                         ← Python sanal ortamı
```

## Kurulum (yeni makine)

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows
# source .venv/bin/activate         # Linux/Mac
pip install -r requirements.txt
```

CUDA için PyTorch'un GPU sürümünü ayrıca kur: https://pytorch.org/

## Çalıştırma

Notebook'u **kök dizinden** açın (yollar `data/...` şeklinde göreli):

```bash
jupyter notebook deeplabv3plus_v9.ipynb
```

Eğitim sonrası Class 4 error analysis:

```bash
python scripts/class4_error_analysis.py
```

> **Windows + Türkçe locale uyarısı:** Script `→` gibi Unicode karakterler içerdiği için `PYTHONIOENCODING=utf-8` set'lenmesi gerekebilir.
