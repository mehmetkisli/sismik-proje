# Kaldığın Yer — Devam Listesi

> **Son senkronizasyon:** 2026-05-17 (Windows PC'de ablation paketi koşturuldu + sunum dosyalarına işlendi)
> **Hedef:** 15 Haziran 2026 final seminer (29 gün kaldı)
> **Bu dosya:** Diğer makinede kaldığın yerden devam ederken ilk açacağın dosya. Diğer her şey bunun pointer'ı.

---

## 🟢 Tamamlandı (bu makinede)

- ✅ Sunum scripti v9 ensemble'a göre baştan revize edildi → [`SUNUM_SCRIPTI.md`](SUNUM_SCRIPTI.md)
  - Single-seed bias bölümü eklendi (28:00–31:00)
  - SFM mini-bölüm eklendi (31:00–33:30)
  - Methodology fix öncesi/sonrası tablosu netleştirildi
  - Beklenen sorular S1–S14'e genişletildi (5 yeni: S10–S14)
- ✅ Kanonik sayılar belgesi → [`SAYILAR_KANONIK.md`](SAYILAR_KANONIK.md)
- ✅ Marp slayt seti (28 slayt) → [`slides_v9_marp.md`](slides_v9_marp.md)
- ✅ Q&A yedek slaytları (15 slayt) → [`slides_v9_backup.md`](slides_v9_backup.md)
- ✅ 4 manuel diyagram üretildi → [`../../results/figures/`](../../results/figures/)
  - `2_5d_input_schema.png`, `methodology_fix_schema.png`, `deeplab_block_diagram.png`, `ensemble_schema.png`
- ✅ Ablation paketi hazırlandı → [`../../ablation/`](../../ablation/)
- ✅ **(2026-05-17) IEEE konferans tezi yazıldı + tam revize edildi** → [`../tez/SeismicFacies_IEEE_Conf_Kisli.tex`](../tez/SeismicFacies_IEEE_Conf_Kisli.tex)
  - Abstract'a FwIoU 0.892 ana sayı eklendi (Alaudah +6.0 puan)
  - Yeni V.H bölümü: SFM ile karşılaştırma (v1/v2 + Class 4 = 0.270 bulgusu)
  - Tartışma'ya Class 4 Test2 confusion tablosu eklendi (%43.9 Under Zechstein'a karışıyor)
  - Sınıf-bazlı IoU tablosu ensemble değerlerine çevrildi (kanonik belgeyle uyumlu)
  - Reproducibility paragrafı SFM dosyaları + FwIoU hesabını dahil edildi
  - QuadrupleLoss ağırlıkları için gerekçe paragrafı eklendi
  - tab:perclass başlık hatası ("v9 tek-seed" diyordu, içerik v7-fixed idi) düzeltildi
  - 10 etiket/ref, 10 table, 11 tabular, 2 figure, 15 cite — hepsi dengeli
  - Bibliografi: UmixClick 2025 referansı eklendi
- ✅ **(2026-05-17) Windows PC ortamı sıfırdan kuruldu** (conda `sismik` env, Python 3.11, PyTorch 2.6+cu124, RTX 4070)
- ✅ **(2026-05-17) Veri Zenodo'dan otomatik indirildi** (`data/train/`, `data/test_once/` hazır)
- ✅ **(2026-05-17) `scripts/eval_originalres.py` script'i YAZILDI** (Madde 0) — `evaluate_ensemble.py` adapte edildi, softmax orijinal çözünürlüğe upsample ediliyor, FwIoU dahil. Koşum yapılamadı (checkpoint transferi yapılmadı).
- ✅ **(2026-05-17) Ablation paketi koşturuldu** (3 training ablation, ~3.5 saat GPU toplam)
  - `--no-lovasz`: Combined 77.92% (Δ +0.23p), C4 Test2 21.61% (Δ −1.43p)
  - `--channels 3`: Combined 77.14% (Δ −0.55p), C4 Test2 11.99% (Δ −11.05p)
  - `--no-xline-aware`: Combined 76.88% (Δ −0.81p), C4 Test2 13.16% (Δ −9.88p)
  - Çıktı: `results/metrics/ablation_summary.md` + `ablation_*_metrics.json`
  - Bulgu: 5-ch 2.5D ve xline-aware aug Class 4 için **olmazsa olmaz** (her biri ~−10p)
- ✅ **(2026-05-17) Ablation sonuçları sunum dosyalarına işlendi** (Madde 2)
  - [`SUNUM_SCRIPTI.md`](SUNUM_SCRIPTI.md) → S5 cevabı gerçek tabloyla güncellendi
  - [`slides_v9_backup.md`](slides_v9_backup.md) → YS-4 tablo TBD'siz, gerçek sayılarla
  - [`SAYILAR_KANONIK.md`](SAYILAR_KANONIK.md) → yeni Madde 10 (Ablation Sonuçları) eklendi
- ⚠️ **(2026-05-17) Mac→PC checkpoint transferi yapılmadı** — bu yüzden Madde 0 (eval_originalres koşumu) ve TTA varyant ablation (Madde 1 #4) atlandı. İkisi de tez/sunumda "future work" olarak kalır (tez Bölüm V.G zaten "pending" diyor, sunum YS-4 not düşüldü).

---

## 🔴 Kalan İşler (öncelik sırasına göre)

### 1. Slayt 2 görseli (manuel, ~15 dk)

Marp slaytında 2. slayt `assets/whatsapp/` placeholder kullanıyor:

```markdown
![bg right:40% fit](../../assets/whatsapp/)  <!-- TODO: sismik kesit + manuel yorum görseli yerleştir -->
```

`assets/whatsapp/` klasöründen uygun görseli seç, dosya adıyla path'i güncelle:

```bash
ls -la assets/whatsapp/   # mevcut görselleri listele
```

Sonra [`slides_v9_marp.md:42`](slides_v9_marp.md) satırını uygun dosya adıyla değiştir.

---

### 2. DeepLab diyagramı rötuş (opsiyonel, ~10 dk)

`results/figures/deeplab_block_diagram.png`'de skip connection oku Concat kutusunun içine çakışıyor. Eğer rahatsız ederse:

```bash
# scripts/generate_presentation_diagrams.py içinde make_deeplab_diagram() fonksiyonu
# Skip connection ok hedefini (8.65, 2.7) → (8.65, 3.1) yap
# Sonra:
venv/bin/python scripts/generate_presentation_diagrams.py
```

---

### 3. Marp slaytları PDF'e çevir (~5 dk)

```bash
# Marp CLI kuruluysa:
npx @marp-team/marp-cli@latest docs/sunum/slides_v9_marp.md --pdf
npx @marp-team/marp-cli@latest docs/sunum/slides_v9_backup.md --pdf

# Veya VS Code Marp extension kullan (gerçek zamanlı önizleme)
```

PDF'ler: `docs/sunum/slides_v9_marp.pdf` + `slides_v9_backup.pdf`

---

### 4. Provalar — 3 tam tekrar (sen yapacaksın, ~40 dk × 3 = 2 saat + analiz)

| # | İş | Hedef | Notlar |
|---|---|---|---|
| 4a | **Prova 1** — tam 40 dk sesli prova, ses kaydı al | Zaman ölçümü, sorunlu geçişleri işaretle | İlk seferde 45+ dk olabilir, sıkıştırma yerleri belirle |
| 4b | Kayıt analizi | Slayt başı zaman bütçesi tablosu | "Burada yavaşladım", "burada nefes almadım" notları |
| 4c | **Prova 2** — revize edilmiş script + slayt | Akıcılık | Q&A senaryosu da dene (S1, S6, S7, S10 sorulacak gibi davran) |
| 4d | **Prova 3** — final | Cila + ezberleme | Anahtar mesajlar 4 madde (script sonunda) sıkı ezberle |

**Önemli:** [`SAYILAR_KANONIK.md`](SAYILAR_KANONIK.md) → "10. KESINLIKLE KARIŞTIRMA Listesi"ni ezberle. Provada yanlış sayı söylemek = sunum sırasında düzeltilmesi zor hata.

---

## 📋 Sunum Günü Hazırlığı (15 Haziran sabahı)

- [ ] PDF'ler indirildi (slides_v9_marp.pdf + slides_v9_backup.pdf)
- [ ] Yedek: USB + cloud + email'e atıldı
- [ ] [`SAYILAR_KANONIK.md`](SAYILAR_KANONIK.md) bir kez daha okundu
- [ ] Su şişesi + temiz gömlek
- [ ] Salon erken kontrol — projektör, klavye, lazer pointer

---

## 🗂️ Yeni / Önemli Dosyalar Referansı

```
docs/sunum/
├── SUNUM_SCRIPTI.md            ← Konuşma metni (40 dk, 28 slayt)
├── SAYILAR_KANONIK.md          ← Tek doğru sayı kaynağı (provada elinin altında)
├── slides_v9_marp.md           ← Ana slayt seti (28 slayt, Marp markdown)
├── slides_v9_backup.md         ← Q&A yedek slaytları (15 slayt)
└── DEVAM_EDILECEK.md           ← BU DOSYA — devam listesi

ablation/
├── README.md                   ← Ablation çalıştırma kılavuzu
├── train_v9_ablation.py        ← Parametreli training (3 ablation flag)
├── eval_tta_variants.py        ← TTA inference ablation
└── merge_ablation_results.py   ← Sonuç → markdown tablo

scripts/
└── generate_presentation_diagrams.py  ← 4 diyagram üreteci (matplotlib)

results/figures/
├── 2_5d_input_schema.png       ← Slayt 9
├── methodology_fix_schema.png  ← Slayt 12-13
├── deeplab_block_diagram.png   ← Slayt 7
└── ensemble_schema.png         ← Slayt 23
```

---

## 🎯 Tek Cümlede Strateji

> "Ablation'ı koştur, sonuçları Claude'a yapıştır, slaytları PDF'e çevir, 3 prova yap. Sunum dürüstlük + multi-seed pedagoji + SFM trade-off üzerine kurulu — bu üçü ezberlendi mi sunum güvenli."

---

## ❓ Diğer Makinede Hatırlatma

- **Branch:** `main` (kontrol et: `git status`)
- **Son commit:** `git log -1 --oneline`
- **Bekleyen değişiklikler:** Ablation script'leri + sunum dosyaları (henüz commit edilmemiş olabilir)
- **Push ettin mi?** `git status` ile kontrol et, gerekirse `git push origin main`

Diğer makinede:

```bash
git pull origin main  # eğer push ettiysen
# veya
git status  # mevcut state'i gör
```

Sunum klasörü değişti — [`SUNUM_SCRIPTI.md`](SUNUM_SCRIPTI.md) ve [`slides_v9_marp.md`](slides_v9_marp.md) yeni sayıları içeriyor. Eski PDF'lere (Nisan tarihli) güvenme.
