# SOTA Literatür Karşılaştırma Tablosu — F3 Hollanda Sismik Fasiyes Segmentasyonu

> **Sunum hazırlığı için referans belge.** Web araştırması sonucu — her sayı kaynak doğrulamalı.
>
> **EN ÖNEMLİ BULGU (sunumda dikkat et):** Alaudah 2019 benchmark'ı **mIoU rapor etmiyor** — bunun yerine **PA (Pixel Accuracy)**, **MCA (Mean Class Accuracy)** ve **FWIU (Frequency-Weighted IoU)** kullanır. Senin v7'nin Combined mIoU 0.793 sayısı, Alaudah'ın FWIU 0.832 sayısıyla **birebir karşılaştırılabilir DEĞİLDİR** — farklı metrikler.
>
> **Aksiyon:** Lokalde v7-fixed modelinden FWIU/PA/MCA hesaplaman lazım, böylece direkt Alaudah Tablo 2'ye satır eklersin. Aksi halde "%79 SOTA mı?" sorusunun cevabı belirsiz kalır.

---

## 1. Metodolojik Uyarı (önce okunmalı)

Alaudah 2019 benchmark'ı F3'ü iki teste böler:
- **Test1:** NW bölgesi (inline 100–299, xline 300–1000) — daha homojen
- **Test2:** E bölgesi (inline 100–700, xline 1001–1200) — büyük Zechstein diapiri içerir

Birçok sonraki makale bu **coğrafi split'i kullanmaz**:
- Wiley 2022 / CONSS 2023 → rastgele 60/20/20 patch split (data leakage olasılığı)
- AdaSemSeg 2025 → kendi F3 train/val/test düzeni
- SFM 2023 → Parihaka'ya geçer (F3 facies değil)

**Sonuç:** Alaudah dışındaki sayıların çoğu **doğrudan karşılaştırılabilir değildir**. Tabloda her satırın "Not" sütununda işaretlendi.

---

## 2. Ana Tablo — F3 Üzerinde Rapor Edilen Sayılar

| # | Yöntem | Yıl | Mimari | Test1 mIoU | Test2 mIoU | Combined / overall | PA | Not |
|---|--------|-----|--------|-----------:|-----------:|--------------------|----|-----|
| 1a | Alaudah patch baseline | 2019 | DeconvNet patch | — | — | FWIU **0.640**, MCA 0.565 | 0.788 | Tablo 2; Test1/Test2 ayrı verilmemiş |
| 1b | Alaudah patch + aug | 2019 | + augmentation | — | — | FWIU 0.743, MCA 0.689 | 0.852 | aynı tablo |
| 1c | Alaudah patch + aug + skip | 2019 | + skip-conn. | — | — | FWIU 0.757, MCA 0.705 | 0.862 | aynı tablo |
| 1d | Alaudah section baseline | 2019 | Section deconvNet | — | — | FWIU 0.789, MCA 0.716 | 0.879 | aynı tablo |
| 1e | Alaudah section + aug | 2019 | Section + aug | — | — | FWIU 0.844, MCA 0.804 | 0.901 | aynı tablo |
| **1f** | **Alaudah section + aug + skip** | **2019** | **Section best baseline** | — | — | **FWIU 0.832, MCA 0.817** | **0.905** | **Alaudah'ın "best baseline"ı** |
| 2 | UmixClick | 2025 | MiT + MSAM, **interactive** | — | — | mIoU 0.7666 | 0.9351 | Nature SR; **interactive (kullanıcı tıklamalı)** — adil değil |
| 3a | AdaSemSeg (5-shot, F3 inline) | 2025 | DGPNet+ResNet50+SimCLR | — | — | FwIoU 0.81, MCA 0.79 | 0.89 | Tablo III; **mIoU değil FwIoU** |
| 3b | AdaSemSeg (5-shot, F3 xline) | 2025 | aynı | — | — | FwIoU 0.80, MCA 0.73 | 0.87 | aynı |
| 3c | AdaSemSeg Baseline-1 (F3 inline) | 2025 | ResNet-UNet target-only | — | — | FwIoU **0.86**, MCA 0.89 | 0.91 | farklı F3 split (kendi düzenleri) |
| 4 | Wiley/Hindawi ensemble (Abid et al.) | 2022 | DeepLabv3+ + SegNet ensemble | — | — | mIoU 0.9392 | 0.9852 | **random patch split**, 7-sınıf — **karşılaştırılamaz** |
| 5 | SFM (Foundation Model) | 2023 | ViT + MAE pretrain | — | — | **F3 facies için rapor edilmedi** (Parihaka 0.798) | — | F3 değil Parihaka kullanmış |
| 6 | CONSS (semi-supervised, %1 etiket) | 2023 | Kontrastif öğrenme | — | — | mIoU 0.9462 (iddia) | — | Alaudah split DEĞİL — **şüpheli yüksek** |
| 7 | Chevitarese et al. (EAGE) | 2018 | CNN patch-based | — | — | "%99 acc" iddiası | ~0.99 PA | **PA, mIoU değil**; Alaudah'tan önce |
| 8 | Liu et al. (Geophysics) | 2020 | 3D CNN + semi-sup. GAN | — | — | rapor edilmemiş (paywall) | — | sayı abstract'ta yok |
| 9 | Civitarese et al. | 2019 | Transposed residual FCN | — | — | mIoU >0.99 (Penobscot) | — | **F3 değil — Penobscot** |
| 10 | Babakhin et al. | 2019 | U-Net ensemble | — | — | — | — | **F3 değil — TGS Salt Kaggle** (farklı görev) |
| 11 | Microsoft seismic-deeplearning | 2020 | UNet/SEResNet/HRNet | — | — | repo'da resmî tablo yok | — | baseline kod var, sayı yok |
| 12 | Dramsch & Lüthje (SEG) | 2018 | Transfer learning, 9 sınıf | — | — | %92 patch-classification | — | **patch sınıflandırma — semantik segmentasyon değil** |
| 13 | Zhao 2018 (SEG) | 2018 | Encoder-decoder section | — | — | metrik abstract'ta yok | — | Alaudah'tan önce |
| **★** | **Bizim v7-broken** (eski) | **2026** | **DeepLabV3+ + EffNet-B4 + 2.5D** | **0.788** | **0.699** | **mIoU 0.793** (FwIoU TBD) | 0.941 | **Alaudah geographic split, methodology bug'lu** |
| **★** | **Bizim v7-fixed** | **2026** | **+ Yol A methodology fix** | **0.7625** | **0.6681** | **mIoU 0.7668 / FwIoU 0.878 / MCA 0.861** | **0.932** | **Alaudah baseline'a yakın değerler ⚠️ 320×320 resized eval** |
| **★** | **Bizim v7-c4fix** | **2026** | **+ 5-ch + Lovász + xline-aware aug** | **0.7902** | **0.6668** | **mIoU 0.7779** | **0.937** | Class 4 paket; Class 4 Test2 0.182 (+%55) |
| **★** | **Bizim v9** (aktif) | **2026** | **+ 384×384 + multi-scale TTA** | **0.7752** | **0.6869** | **mIoU 0.7769 / Dice 0.868 / MCA 0.863** | **0.935** | Class 4 Test2 0.230 (+%96 vs v7-fixed) ⚠️ 384×384 eval |
| **★** | **Bizim SFM v1** | **2026** | **SFM ViT-B/16 + 5→1 ch + MLAHead, encoder lr 5e-5** | **0.7209** | **0.6665** | **mIoU 0.7281 / Dice 0.830** | **0.922** | **F3 Alaudah'ta SFM ilk fine-tune.** Class 4 Test2 **0.270** — tüm sürümler arasında en iyi |
| **★** | **Bizim SFM v2** | **2026** | **+ enc lr 2e-4 + layer_decay 0.9 + rare 15** | **0.7823** | **0.6591** | **mIoU 0.7677 / Dice 0.858** | **0.936** | Best val mIoU **0.8233** (en iyi); Test1 v9'u geçti; ama Class 4 Test2 düştü (0.20) — agresif enc fine-tune'un trade-off'u |

---

## 3. Sunum Slide'ı için "Tek Tablo" Özet (sayfayı tek bir slayta sığdır)

| Yöntem | Yıl | F3'te en yüksek | Split | Eval | Karşılaştırılabilir mi? |
|--------|-----|-----------------|-------|------|--------------------------|
| Alaudah section + aug + skip (baseline) | 2019 | PA **0.905**, FwIoU **0.832**, MCA **0.817** | Test1+Test2 (geographic) | Orijinal çözünürlük | ✅ Aynı split |
| **Bizim v9 (DeepLabV3+ + EffNet-B4 + 5-ch 2.5D + Lovász + multi-scale TTA)** | **2026** | mIoU **0.777** / **PA 0.935** / **MCA 0.863** | Alaudah geographic (methodology-fixed) | **384×384 resized** | ⚠️ Aynı split, **farklı evaluator çözünürlüğü** — birebir kıyas için orijinal-resolution evaluator gerekli |
| **Bizim v7-fixed (referans baseline)** | **2026** | mIoU **0.767** / **PA 0.932** / **FwIoU 0.878** / **MCA 0.861** | Alaudah geographic (methodology-fixed) | **320×320 resized** | ⚠️ Aynı koşullar |
| AdaSemSeg Baseline-1 (target-only) | 2025 | F3 inline FwIoU **0.86**, PA **0.91** | farklı F3 split | farklı | ⚠️ farklı split |
| UmixClick (interactive) | 2025 | mIoU **0.7666**, PA **0.9351** | belirsiz, **kullanıcı yardımı** | belirsiz | ❌ adil değil |
| Wiley/Hindawi ensemble | 2022 | mIoU **0.9392**, PA **0.9852** | random 60/20/20, **7-sınıf** | belirsiz | ❌ farklı problem |

---

## 4. Bizim Sayıların Literatürdeki Yeri (Sunum Yorumu)

### 4.1 Net Bulgular (v7-fixed sonuçları geldikten sonra)

- **v7-fixed sayıları Alaudah baseline'a yakın değerlerde görünüyor** (Combined):
  - PA: 0.932 vs Alaudah 0.905 → **+2.7 puan**
  - MCA: 0.861 vs Alaudah 0.817 → **+4.5 puan**
  - FwIoU: 0.878 vs Alaudah 0.832 → **+4.6 puan**
- ⚠️ **ÖNEMLİ KAVET:** Bizim sayılarımız 320×320 resized space'te hesaplandı; Alaudah orijinal çözünürlükte (701×255 / 401×255) evaluator kullanır. **"Geçtik" iddiasını yapmadan önce aynı evaluator protokolünü çalıştırmak gerekir** — bu original-resolution evaluator henüz yazılmadı.
- Şu anki dürüst ifade: **"Alaudah baseline seviyesinde, methodology hataları düzeltilmiş, dürüstçe rapor edilmiş bir DeepLabV3+ baseline"**. Original-resolution evaluator yapıldıktan sonra bu kıyas kesinleşecek.
- mIoU sayımız (0.767) "düşük" görünür ama Alaudah bu metriği rapor etmemiştir — bu yöndeki kıyas zaten geçersiz.
- AdaSemSeg Baseline-1 (target-only, F3 inline FwIoU 0.86) bize en yakın kıyasdır; bizim Combined FwIoU 0.878 ile yakın bantta ama AdaSemSeg farklı F3 split'i kullandığı için doğrudan kıyas tartışmalı.
- Wiley 2022 / CONSS 2023'ün 0.94+ mIoU sayıları farklı split + farklı sınıf + data leakage ile şişirilmiş — bizim sayılarımızla doğrudan karşılaştırılamaz.

### 4.2 Sunum İçin Dürüst İfadeler

> *"DeepLabV3+ + EfficientNet-B4 + 2.5D modelimiz, Alaudah 2019 geographic split'ine sadık kalan az sayıda yöntemden biridir. Combined mIoU 0.77 sayımız Alaudah'ın 'best baseline'ı (FwIoU 0.832, PA 0.905) ile birebir kıyaslanabilir değil — farklı metrikler. Aynı metrikleri (FwIoU, PA, MCA) lokal modelimizden hesapladığımızda baseline'a yakın değerler elde ediyoruz, ancak evaluator çözünürlük farkı (320×320 vs orijinal) nedeniyle 'geçtik' iddiasını yapmıyoruz — original-resolution evaluator bekleyen iş."*

> *"Literatürde rapor edilen 0.94+ mIoU sayıları (Wiley 2022, CONSS 2023) genellikle rastgele patch split veya farklı sınıf bölünmesi kullanır — Alaudah'ın orijinal coğrafi split'i değil. Bu yüzden bu sayılar bizim 0.77'mizle doğrudan karşılaştırılamaz."*

> *"Mevcut çalışma SOTA iddiasında değil. Modern literatür (2025-2026) sismik-spesifik foundation model'lara (SFM, GFM) doğru evrildi. Bizim ImageNet pretrained EfficientNet-B4 yaklaşımımız bu perspektifte baseline seviyesinde, methodology titizliği ile değerlendirilen bir DeepLabV3+ baseline'ı sunar."*

### 4.3 Eğer Test1/Test2 ayrımı sorulursa

Alaudah 2019 makalesinin **Tablo 2'sinde Test1/Test2 ayrı sayılar yok**. Yazarlar bit.ly link'iyle ayrı paylaşmış olabilir ama makalede sadece birleşik sayı var. Bu, Test1/Test2 ayrımını rapor eden bir referans yöntemin literatürde nadir olduğunu gösterir — bizim Test1 0.788 / Test2 0.699 ayrı raporlamamız aslında yöntemsel bir **artı**.

---

## 5. ⚠️ "GÜVENİLİR DEĞİL" Listesi (sunumda kullanırken çekince koy)

Bu sayılar tabloda görünebilir ama jüri sorduğunda "metodolojik fark var" demeye hazır ol:

| Kaynak | Sayı | Sorun |
|--------|------|-------|
| Wiley/Hindawi 2022 | mIoU 0.9392 | Random patch split, 7 sınıf, data leakage olasılığı |
| CONSS 2023 | mIoU 0.9462 | Alaudah split DEĞİL, kendi train/test düzeni |
| Civitarese 2019 | mIoU >0.99 | F3 değil — **Penobscot** |
| Chevitarese 2018 | "%99 accuracy" | PA, mIoU değil — Alaudah'tan önce |
| Liu et al. 2020 F3 sayıları | — | Paywall, abstract'ta yok |
| Dramsch & Lüthje 2018 | "%92" | Patch sınıflandırma, semantik segmentasyon değil |
| SFM F3 facies | — | F3 değil **Parihaka** kullanmış |
| Microsoft seismic-deeplearning | — | Resmî SOTA tablosu doğrulanamadı |
| Babakhin 2019 | — | F3 değil **TGS Salt** — listeye dahil edilmemeli |
| Pham & Fomel 2024 attention U-Net + freq | — | Spesifik makale aramayla doğrulanamadı |

---

## 6. Aksiyon Önerisi (eğitim biter bitmez)

v7-fixed modelinden **FWIU, PA, MCA** metriklerini hesapla — Alaudah Tablo 2 ile birebir karşılaştırılabilir tablo üret:

```python
# notebook'ta cell-22'ye veya yeni hücreye ekle:
def compute_fwiu(preds, targets, n=NUM_CLASSES):
    """Frequency-Weighted IoU (Alaudah 2019 ile aynı formül)."""
    ious = []; freqs = []
    for c in range(n):
        gt_c = (targets == c)
        pred_c = (preds == c)
        inter = (gt_c & pred_c).sum()
        union = (gt_c | pred_c).sum()
        ious.append(inter / (union + 1e-8))
        freqs.append(gt_c.sum() / targets.size)
    return float(np.average(ious, weights=freqs))

# combined için:
fwiu_combined = compute_fwiu(np.concatenate([p1, p2]), np.concatenate([t1, t2]))
print(f"Combined FWIU: {fwiu_combined:.4f}  (Alaudah best: 0.832)")
```

`compute_metrics` zaten `MCA` ve `PA` hesaplıyor — sadece `FWIU` eklemek yeter. Bu sayıyla Alaudah Tablo 2'ye direkt satır eklersin:

| Yöntem | FWIU | PA | MCA |
|--------|------|----|----|
| Alaudah section + aug + skip | 0.832 | 0.905 | 0.817 |
| **Bizim v7-fixed** | **TBD** | **TBD** | **TBD** |

---

## 7. Tam Kaynak Linkleri

- Alaudah benchmark — [arXiv:1901.07659](https://arxiv.org/abs/1901.07659), [GitHub](https://github.com/yalaudah/facies_classification_benchmark), [SEG Interpretation](https://library.seg.org/doi/10.1190/int-2018-0249.1)
- AdaSemSeg — [arXiv:2501.16760](https://arxiv.org/abs/2501.16760)
- Seismic Foundation Model (SFM) — [arXiv:2309.02791](https://arxiv.org/abs/2309.02791), [GitHub](https://github.com/shenghanlin/SeismicFoundationModel)
- ThinkOnward GFM — [HuggingFace](https://huggingface.co/thinkonward/geophysical-foundation-model), [GitHub](https://github.com/thinkonward/geophysical-foundation-model)
- UmixClick (Nature SR 2025) — [doi:10.1038/s41598-025-32016-8](https://www.nature.com/articles/s41598-025-32016-8)
- Wiley/Hindawi ensemble — [doi:10.1155/2022/7762543](https://onlinelibrary.wiley.com/doi/10.1155/2022/7762543)
- CONSS — [arXiv:2210.04776](https://arxiv.org/abs/2210.04776), [IEEE JSTARS](https://ieeexplore.ieee.org/document/10230299/)
- Liu et al. 2020 — [doi:10.1190/geo2019-0627.1](https://library.seg.org/doi/10.1190/geo2019-0627.1)
- Civitarese 2019 — [arXiv:1905.04307](https://arxiv.org/abs/1905.04307)
- Chevitarese 2018 EAGE — [doi:10.3997/2214-4609.201800237](https://www.earthdoc.org/content/papers/10.3997/2214-4609.201800237)
- Dramsch & Lüthje 2018 — [doi:10.1190/segam2018-2996783.1](https://library.seg.org/doi/10.1190/segam2018-2996783.1)
- Microsoft seismic-deeplearning — [GitHub](https://github.com/microsoft/seismic-deeplearning)

---

## Son Notlar

- Alaudah 2019 PDF Tablo 2 detayları doğrulandı (sayılar yukarıda)
- AdaSemSeg PDF Tablo I, II, III doğrulandı (sayılar yukarıda)
- SFM'in F3 facies için sayı raporlamadığı doğrulandı (yaygın yanılgı)
- Wiley 2022, CONSS 2023, Civitarese 2019 sayıları metodolojik olarak farklı — sunumda dürüstçe işaretle
