# v9 Ablation Paketi

> Sunum için ablation tablosu üretir. v9 mimarisinin 4 bileşenini tek tek devre dışı bırakıp katkılarını ölçer.
>
> **GPU gerekli.** Toplam tahmini süre: ~10-14 saat RTX 3060 Ti.

---

## Neyi Test Ediyoruz?

v9 = DeepLabV3+ + EffNet-B4 + **5-ch 2.5D** + **QuadrupleLoss (Lovász içerir)** + **xline-aware aug** + **multi-scale TTA** + 384×384 + Yol A methodology fix + Mixup + WeightedSampler.

Hangi bileşen ne kadar katkı yapıyor?

| # | Ablation | Bileşen | Maliyet | Beklenti |
|---|---|---|---|---|
| 1 | `--no-lovasz` | Lovász loss OFF (TripleLoss) | ~2.5 saat | -0.5 ile -1.0 puan Combined mIoU |
| 2 | `--channels 3` | 3-channel 2.5D (±1 yerine ±2) | ~2.5 saat | Combined küçük etki, C4 Test2'de büyük düşüş |
| 3 | `--no-xline-aware` | xline aug = inline aug | ~2.5 saat | Test2 mIoU -1 ile -2 puan |
| 4 | TTA varyantları | no-TTA / hflip / multi-scale | ~30 dk (inference) | -1 ile -2 puan no-TTA vs default |

**Eğitim toplamı:** ~7.5 saat (3 ablation × 2.5 saat).
**Inference toplamı:** ~30 dk.
**Genel toplam:** ~8 saat.

---

## Önkoşullar

1. `venv` aktif olmalı (PyTorch + smp + albumentations kurulu)
2. **v9 baseline checkpoint** olmalı: `checkpoints_v7/deeplabv3plus_v9_best.pth` (TTA ablation için)
3. Veri: `data/train/`, `data/test_once/`
4. Sonuçlar: `results/metrics/ablation_*.json` yazılır

---

## Çalıştırma Komutları (sırayla, copy-paste)

### Mac / Linux

```bash
# Hızlı sanity-check (~5-10 dk, eğitim doğru başlıyor mu kontrol et)
PYTHONIOENCODING=utf-8 venv/bin/python ablation/train_v9_ablation.py --no-lovasz --epochs 3

# 1) Lovász OFF (TripleLoss) — ~2.5 saat
PYTHONIOENCODING=utf-8 venv/bin/python ablation/train_v9_ablation.py --no-lovasz

# 2) 3-channel 2.5D — ~2.5 saat
PYTHONIOENCODING=utf-8 venv/bin/python ablation/train_v9_ablation.py --channels 3

# 3) xline-aware aug OFF — ~2.5 saat
PYTHONIOENCODING=utf-8 venv/bin/python ablation/train_v9_ablation.py --no-xline-aware

# 4) TTA varyantları (inference only) — ~30 dk
PYTHONIOENCODING=utf-8 venv/bin/python ablation/eval_tta_variants.py

# 5) Sonuçları tabloya çevir (markdown + json)
venv/bin/python ablation/merge_ablation_results.py
```

### Windows (PowerShell)

```powershell
# Hızlı sanity-check
$env:PYTHONIOENCODING="utf-8"; .\venv\Scripts\python.exe ablation\train_v9_ablation.py --no-lovasz --epochs 3

# 1-3) Training ablation'ları
.\venv\Scripts\python.exe ablation\train_v9_ablation.py --no-lovasz
.\venv\Scripts\python.exe ablation\train_v9_ablation.py --channels 3
.\venv\Scripts\python.exe ablation\train_v9_ablation.py --no-xline-aware

# 4) TTA varyant ablation
.\venv\Scripts\python.exe ablation\eval_tta_variants.py

# 5) Sonuçları birleştir
.\venv\Scripts\python.exe ablation\merge_ablation_results.py
```

---

## Önemli Notlar

### Resume desteği
- Her ablation kendi checkpoint'ini yazıyor: `checkpoints_v7/ablation_{TAG}_checkpoint.pth`
- Kesintide aynı komutu yeniden çalıştırırsan kaldığı yerden devam eder
- Best model: `checkpoints_v7/ablation_{TAG}_best.pth`

### Disk alanı
- Her checkpoint ~343 MB (model + optimizer + scheduler state)
- 3 ablation = ~1 GB checkpoint + 1 GB best model = **~2 GB**
- TTA ablation disk yazmıyor (sadece JSON)

### Seed = 42
- Tüm ablation'lar SEED=42 (v9 ana ile aynı) → apples-to-apples
- Multi-seed ablation **YAPILMIYOR** — single-seed varyansı zaten 3-seed ensemble'da raporlandı
- Ablation tek-seed: bilinen sınırlılık, sunumda "tek-seed ablation" diye söylenecek

### Eğitim erken biterse?
- `--epochs N` ile epoch sayısı azaltılabilir (örn: `--epochs 50`)
- Hızlı bir gauge için `--epochs 30` yeterli olabilir — early stopping zaten ~50-60 civarı tetikleniyor

---

## Sonuçları Sunuma Aktarma

`merge_ablation_results.py` çalıştırıldıktan sonra:

1. **`results/metrics/ablation_summary.md`** dosyası üretilir
2. Bu markdown tablosunu **iki yere kopyala**:
   - [`docs/sunum/SUNUM_SCRIPTI.md`](../docs/sunum/SUNUM_SCRIPTI.md) — Soru S5 cevabı (ablation tablosu yorumu)
   - [`docs/sunum/slides_v9_backup.md`](../docs/sunum/slides_v9_backup.md) — YS-4 yedek slaydı (sayıları güncelle)

3. Ayrıca [`docs/sunum/SAYILAR_KANONIK.md`](../docs/sunum/SAYILAR_KANONIK.md) → "9. Eksik Sayılar" bölümünden ablation TBD satırını çıkar, gerçek sayıları doldur.

---

## Beklenmedik Durumlar

### OOM (out of memory) hatası
- Batch boyutunu azalt: kod içinde `BATCH_SIZE = 4` → `BATCH_SIZE = 2`, `ACCUM_STEPS = 6` → `ACCUM_STEPS = 12` (efektif batch sabit kalır)
- Veya çözünürlüğü düşür: `IMG_SIZE = (384, 384)` → `(320, 320)` (v9 baseline'ı 384, kıyas tutarsızlığı doğurur — son çare)

### NaN loss
- Lovász + AMP fp16 NaN üretebilir — kod zaten FP32 wrap içeriyor, sorun olmamalı
- Olursa: `use_amp = False` ile koştur (eğitim 2x yavaşlar)

### Çok yavaş?
- `NUM_WORKERS = 2` Mac/Linux için, Windows'ta 0 (multiprocessing sorunu)
- `pin_memory=True` GPU varsa otomatik açık

---

## Sonuçlar Geldiğinde

Bana sadece şunu söylemen yeterli:

```
ablation/merge_ablation_results.py çalıştırdım, sonuçlar:
[ablation_summary.md içeriğini buraya yapıştır]
```

Ben de sayıları script + slayda işlerim, ablation hikâyesini sunum içine yerleştiririm.
