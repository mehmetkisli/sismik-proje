# Kanonik Sayılar — Sunum Referansı

> **Sunum + tez + tüm dokümanlarda tek doğru kaynak.** Ana model = **v9 Multi-seed Ensemble** (SEED 42+43+44 softmax averaging). Sunumda söylenen her sayı bu dosyaya uymak zorunda.
>
> **Son güncelleme:** 2026-05-12

---

## 1. Ana Sayı (sunum başlığı / abstract / sonuç slaydı)

> **Combined mIoU = 0.791 ± 0.006** (3-seed ensemble, multi-scale TTA, 384×384 resized eval, Yol A methodology fix)

Tek bir cümlede söylemek gerekirse, **bu** sayı tezin ve sunumun ana sonucudur. Diğer her metrik bunu destekler.

---

## 2. v9 Ensemble — Tam Metrik Tablosu

| Metrik | Test1 (inline) | Test2 (xline) | **Combined** |
|---|---:|---:|---:|
| mIoU | 0.8043 | 0.6771 | **0.7910** |
| Mean Dice | 0.8858 | 0.7717 | 0.8769 |
| Pixel Accuracy (PA) | 0.9383 | 0.9422 | 0.9402 |
| Mean Class Accuracy (MCA) | 0.9007 | 0.7525 | 0.8734 |

Kaynak: [results/metrics/v9_ensemble_metrics.json](../../results/metrics/v9_ensemble_metrics.json)

---

## 3. Seed Varyansı (3-seed)

| SEED | Combined mIoU | Best val mIoU |
|---|---:|---:|
| 42 (v9 ana) | 0.7769 | 0.8022 |
| 43 | 0.7850 | 0.8271 |
| 44 | 0.7920 | 0.8078 |
| **Mean ± Std** | **0.7846 ± 0.0062** | 0.8124 ± 0.013 |
| **Ensemble (softmax avg)** | **0.7910** | — |

**Kritik:** Ensemble (0.791), tek tek seed'lerin ortalamasından (0.785) yüksek — ensembling'in net katkısı **+0.6 puan**.

---

## 4. Per-Class IoU (Ensemble Combined)

| Sınıf | İsim | IoU | Test1 | Test2 |
|---|---|---:|---:|---:|
| 0 | Upper North Sea | 0.951 | 0.953 | 0.949 |
| 1 | Lower North Sea | 0.812 | 0.809 | 0.815 |
| 2 | Rijnland | 0.941 | 0.941 | 0.941 |
| 3 | Scruff | 0.629 | 0.674 | 0.562 |
| 4 | **Zechstein** | 0.803 | 0.843 | **0.183** |
| 5 | Under Zechstein | 0.609 | 0.606 | 0.612 |

**Kritik gözlem:** Class 4 Test2 = **0.183** — modelin tek belirgin zayıf noktası.

---

## 5. Single-seed Bias Bulgusu (sunumun en güçlü pedagojik hikâyesi)

Class 4 Test2 IoU:

| SEED | Class 4 Test2 IoU |
|---|---:|
| 42 (v9 ana, ilk raporlanan) | **0.2304** ← outlier-pozitif |
| 43 | 0.1560 |
| 44 | 0.1625 |
| **Mean ± Std** | **0.183 ± 0.04** |
| Ensemble | 0.1834 |

**Akademik mesaj:** "v9 tek-seed 0.230 sonucu **şanslı çıkıştı** — 3-seed gerçek değer 0.183 ± 0.04. Bu, segmentation literatüründe tek-seed raporlamanın risk taşıdığının somut kanıtıdır."

---

## 6. Sürüm Evrimi (Combined mIoU)

| Sürüm | Combined mIoU | Δ | Önemli yenilik |
|---|---:|---:|---|
| v3 (baseline U-Net) | 0.4012 | — | Plain U-Net |
| v5 | 0.7582 | +35.7p | EffNet-B4 + DeepLabV3+ + 320×320 |
| v6 | 0.7580 | -0.0 | + 2.5D + WeightedSampler |
| v7-broken | 0.7926 | (+3.5p — leakage'lı) | (⚠️ methodology bug'lı) |
| **v7-fixed** | **0.7668** | **−2.6p düzeltme** | **+ Yol A methodology fix** |
| v7-c4fix | 0.7779 | +1.1p | + 5-ch + Lovász + xline-aware aug |
| v9 (single-seed) | 0.7769 | -0.1p | + 384×384 + multi-scale TTA |
| **v9 ensemble** | **0.7910** | **+1.4p** | **+ 3-seed softmax averaging** |

**Anlatım:** "v7-broken 0.7926 görünüyor ama leakage'lı yapay sayı. Methodology fix sonrası gerçek baseline 0.767 olarak ortaya çıktı, oradan v9 ensemble ile 0.791'e ulaştık — gerçek bir +2.4p ilerleme."

---

## 7. SFM Deneyleri (mini-bölüm sayıları)

| Model | Combined mIoU | Test1 | Test2 | **Class 4 Test2** | Best val |
|---|---:|---:|---:|---:|---:|
| v9 ensemble (ana model) | **0.7910** | 0.8043 | 0.6771 | 0.183 | 0.8124 |
| SFM v1 | 0.7281 | 0.7209 | 0.6665 | **0.270** ← lider | 0.81 |
| SFM v2 | 0.7677 | 0.7823 | 0.6591 | 0.20 | **0.8233** ← lider |

**Mesaj:** "SFM sayısal olarak ana modeli geçemedi (0.728/0.768 vs 0.791) ama Class 4 Test2'de 0.270 ile **literatürde değerli bir nokta** ürettik — domain-spesifik pretrain'in zorlu sınıfta avantaj sağladığının kanıtı."

---

## 8. Literatür Karşılaştırma (TEK SLAYT için özet)

| Yöntem | mIoU | FwIoU | PA | MCA | Karşılaştırılabilir? |
|---|---:|---:|---:|---:|---|
| Alaudah 2019 section+aug+skip (best baseline) | — | 0.832 | 0.905 | 0.817 | ✅ Aynı split, orijinal eval |
| **v9 ensemble (bizim)** | **0.7910** | TBD | **0.9402** | **0.8734** | ⚠️ Aynı split, 384×384 resized eval |
| v7-fixed (referans baseline) | 0.7668 | 0.878 | 0.932 | 0.861 | ⚠️ Aynı split, 320×320 resized eval |
| AdaSemSeg target-only | — | 0.86 | 0.91 | 0.89 | ⚠️ Farklı F3 split |
| UmixClick (interactive) | 0.7666 | — | 0.9351 | — | ❌ Interactive, adil değil |
| Wiley 2022 ensemble | 0.9392 | — | 0.9852 | — | ❌ Random split + 7 sınıf |

**Savunma cümlesi (ezberlenecek):** "0.94+ mIoU sayılarının çoğu (Wiley 2022, CONSS 2023) **random patch split + farklı sınıf bölünmesi** kullanır — Alaudah'ın orijinal coğrafi split'ine sadık kalan az sayıda yöntemden biriyiz. PA 0.940 / MCA 0.873 değerlerimiz Alaudah baseline'ı (PA 0.905 / MCA 0.817) seviyesinde. Doğrudan 'geçtik' iddiası için original-resolution evaluator gerekli — bu future work."

---

## 9. Eksik Sayılar (TBD — sunum öncesi tamamlanmalı)

- **v9 ensemble FwIoU** — JSON'da yok, hesaplanmalı. Test-set sınıf frekansları × per-class IoU formülüyle çıkar. Tahmini değer: ~0.89-0.90 bandı (v7-fixed 0.878'den yukarı).
- **Ablation tablosu** — Lovász/5-ch/xline-aware aug/TTA ON/OFF sayıları (GPU işleri tamamlanınca buraya eklenecek).
- **Original-resolution evaluator sayıları** — Alaudah birebir kıyas için (future work — sunumdan önce yetişmez, dürüstçe söylenecek).

---

## 10. KESINLIKLE KARIŞTIRMA Listesi

Sunumda **yanlışlıkla** söylenmemesi gereken sayılar:

| Yanlış | Doğru |
|---|---|
| "Combined mIoU 0.793" | 0.793 = v7-**broken** (leakage'lı). Doğrusu **0.791 ensemble** |
| "Combined mIoU 0.767" | 0.767 = v7-fixed (önceki). Doğrusu **0.791 ensemble** |
| "Class 4 Test2 0.230" | Tek-seed outlier. Doğrusu **0.183 ± 0.04 ensemble** |
| "Alaudah'ı geçtik" | Original-eval yok, **"baseline seviyesinde, dürüst rapor"** |
| "SOTA aldık" | SOTA değil — **"methodology rigor + dürüst baseline"** |
