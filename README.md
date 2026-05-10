# Sismik Fasiyes Sınıflandırması — DeepLabV3+

Bu proje, F3 Hollanda sismik veri kümesi (Alaudah 2019 benchmark) üzerinde DeepLabV3+ (EfficientNet-B4) tabanlı semantik segmentasyon ile fasiyes sınıflandırması yapar.

## Aktif Sürüm: v7-fixed

v7-fixed = v7 + Yol A methodology fix (orta blok val + ±2 inline buffer + crossline cropping ile leakage giderme)

| Metrik | Test1 (inline) | Test2 (crossline) | **Combined** |
|---|---:|---:|---:|
| mIoU | 76.25% | 66.81% | **76.68%** |
| Mean Dice | 85.60% | 75.70% | **86.03%** |
| Pixel Accuracy | 92.28% | 94.10% | **93.19%** |
| Mean Class Acc | 87.43% | 74.93% | **86.15%** |
| FwIoU | 86.34% | 89.63% | **87.84%** |

> **Önemli kavet:** Yukarıdaki sayılar 320×320 resized space'te hesaplanmıştır. Alaudah benchmark'ı orijinal çözünürlükte (701×255 / 401×255) evaluator çalıştırır; dolayısıyla **doğrudan kıyaslanabilirlik için tam protokol uyumu yok** — sayılar Alaudah'ın baseline değerlerine yakın bantta ama "geçtik" iddiası için orijinal-çözünürlük evaluator gerekir (bekleyen iş).

> **v7-broken (eski, bug'lı):** Test1 78.81%, Test2 69.86%, Combined 79.26%. Bu sayılar contiguous-block val + 3D crossline leakage + 2.5D komşu sızıntısından beslenen yapay yüksek değerlerdi. v7-fixed'in daha düşük gözükmesi gerçek genelleme performansının ortaya çıkması demek (val-test paradoksu çözüldü).

v7-fixed = EfficientNet-B4 + 2.5D Multi-View + LS-CE+Dice+Focal + Mixup + TTA + methodology fix

Sürüm evrimi: `results/sonuc.txt` ve `results/metrics/*.json`

## Bilinen Sınırlılıklar

- **Class 4 (Zechstein) Test2 IoU 0.118** — model %58.8 oranında Under Zechstein ile karıştırıyor. Tuz tabakasının yön bağımlı (anisotropic) morfolojisi; eğitim inline-baskın temsil. Çözüm: domain adaptation (future work).
- **Tek seed (42)** — error bar yok, varyans ölçülmedi. Multi-seed planlandı.
- **Ablation eksik** — sadece TTA on/off (+0.5 puan); Mixup, 2.5D, xline, rare sampler için ablation yok.
- **Evaluator resize'a bağlı** — 320×320 metrikler. Original-resolution evaluator henüz yok.

Detaylı liste: [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

## Klasör Yapısı

```
.
├── deeplabv3plus_v7.ipynb     ← Aktif eğitim notebook'u (v7-fixed)
├── view_data.ipynb             ← Veri inceleme
│
├── archive/                    ← Eski sürümler (v5, v5.3, v6, v7.0)
│
├── data/                       ← Veri seti (notebook bu yoldan okuyor)
│   ├── train/
│   │   ├── train_seismic.npy   (547 MB)
│   │   └── train_labels.npy
│   └── test_once/
│       ├── test1_seismic.npy
│       ├── test1_labels.npy
│       ├── test2_seismic.npy
│       └── test2_labels.npy
│
├── results/
│   ├── sonuc.txt               ← Sürüm karşılaştırma özeti
│   ├── metrics/                ← v3/v5/v6/v7-broken/v7-fixed JSON metrikleri
│   └── figures/                ← Eğitim/test/error_analysis görselleri
│
├── scripts/
│   └── class4_error_analysis.py  ← Class 4 hata analiz scripti
│
├── docs/
│   ├── PROJE_OZETI.md          ← Detaylı devir belgesi
│   ├── AI_HANDOFF.md           ← Kısa devir notu
│   ├── LIMITATIONS.md          ← Bilimsel sınırlılıklar
│   ├── literature_table.md     ← SOTA karşılaştırma tablosu
│   ├── tez/                    ← LaTeX kaynağı + PDF çıktıları
│   ├── sunum/                  ← Sunum PDF'leri + SUNUM_SCRIPTI.md
│   └── plan/                   ← Proje planı + literatür taraması
│
├── assets/whatsapp/            ← Ekran görüntüleri / fotoğraflar
│
└── venv/                       ← Python sanal ortamı
```

## Kurulum (yeni makine)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

CUDA için PyTorch'un GPU sürümünü ayrıca kur: https://pytorch.org/

## Çalıştırma

Notebook'u **kök dizinden** açın (yollar `data/...` şeklinde göreli):

```bash
jupyter notebook deeplabv3plus_v7.ipynb
```

Eğitim sonrası Class 4 error analysis:

```bash
python scripts/class4_error_analysis.py
```
