# Sismik Fasiyes Sınıflandırması — DeepLabV3+

Bu proje, F3 Hollanda sismik veri kümesi üzerinde DeepLabV3+ (EfficientNet-B4) tabanlı semantik segmentasyon ile fasiyes sınıflandırması yapar.

## Aktif Sürüm: v7

En iyi sonuçlar **v7** modelinden alındı:

| Metrik | Değer |
|---|---|
| Test1 mIoU (inline) | **78.81%** |
| Test2 mIoU (crossline) | 69.86% |
| Combined mIoU | **79.26%** |
| Combined Dice | **87.75%** |
| Combined PA | **94.13%** |

v7 = EfficientNet-B4 + 2.5D Multi-View + Focal+Dice + TTA

Sürüm karşılaştırması: [results/sonuc.txt](results/sonuc.txt) ve [results/metrics/](results/metrics/)

## Klasör Yapısı

```
.
├── deeplabv3plus_v7.ipynb     ← Çalıştır: aktif eğitim notebook'u
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
│   └── metrics/                ← v5/v6/v7 JSON metrikleri
│
├── docs/
│   ├── tez/                    ← LaTeX kaynağı + PDF çıktıları
│   ├── sunum/                  ← Sunum PDF'leri
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
pip install torch segmentation-models-pytorch albumentations numpy matplotlib scikit-learn
```

CUDA için PyTorch'un GPU sürümünü ayrıca kur: https://pytorch.org/

## Çalıştırma

Notebook'u **kök dizinden** açın (yollar `data/...` şeklinde göreli):

```bash
jupyter notebook deeplabv3plus_v7.ipynb
```
