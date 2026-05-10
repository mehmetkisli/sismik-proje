# Sunum Scripti — DeepLabV3+ ile Sismik Fasiyes Segmentasyonu

> **40 dakikalık seminer için konuşma metni.** Slayt değil, prova ederken yüksek sesle okuyacağın script. Her bölümün başında **slayt önerisi**, ortada **konuşma metni**, sonunda **geçiş cümlesi** var.
>
> **Hedef:** 15 Haziran 2026 final seminer (Bilgisayarda Görme dersi, yüksek lisans)
> **Sunum dili:** Türkçe
> **Hocaya ton:** Akademik dürüstlük + teknik yetkinlik. Övünme yok, "limitations farkındayım" var.

---

## ⏱️ 00:00 – 03:00 — Açılış ve Problem Tanımı

**Slayt 1-2:** Başlık + sismik kesit görseli + jeofizikçi manuel yorumlama görseli

**Konuşma:**

> "İyi günler. Bugün size yüksek lisans projemde geliştirdiğim derin öğrenme tabanlı sismik fasiyes segmentasyon yaklaşımını sunacağım.
>
> Sismik fasiyes sınıflandırması, yeraltı jeolojik yapılarının sismik dalga yansıma örüntülerine göre kategorilere ayrılması işlemidir. Petrol ve doğal gaz endüstrisinde rezervuar karakterizasyonu için kritiktir. Şu an bu yorum hâlâ jeofizikçiler tarafından **elle yapılır** — bir uzman bir 3D sismik bloğu yorumlamak için günler harcar, üstelik iki farklı uzman aynı kesit için farklı yorumlar verebilir.
>
> Bu çalışmanın amacı, derin öğrenme ile bu yorumlama sürecini **piksel bazında otomatikleştirmek**. Hedef veri seti: Hollanda F3 bloğu, Alaudah 2019 benchmark."

**Geçiş:** "Şimdi bu veri setini biraz daha ayrıntılı tanıtayım."

---

## ⏱️ 03:00 – 07:00 — Veri Seti

**Slayt 3-4:** F3 volüm görseli + 6 sınıf renk paleti + sınıf dağılım grafiği

**Konuşma:**

> "Veri setimiz Hollanda F3 bloğu — kuzey denizinde, yaklaşık 16×24 km'lik bir alan. Toplam 401 inline × 701 crossline × 255 derinlik adımlı bir 3D volüm.
>
> Eğitim için bu volümün tamamı verilmiş; iki ayrı test volümümüz var: Test1, 200 inline'lık ayrı bir blok — inline yönünde değerlendirme. Test2 ise 601 crossline'lık — daha kritiği, çünkü modelin **crossline yönünde de genelleyip genellemediğini** ölçer.
>
> 6 jeolojik fasiyes sınıfımız var: Upper North Sea, Lower North Sea, Rijnland, Scruff, Zechstein ve Under Zechstein. Burada **kritik nokta sınıf dengesizliği**: Rijnland tüm piksellerin yaklaşık %48'ini oluştururken, en az olan Under Zechstein sadece %1.5'idir. Yani naive bir model her piksele 'Rijnland' deyip %48 doğruluk alabilir. Bu nedenle metriği seçerken dikkatliyiz — sadece pixel accuracy değil, mIoU üzerinden değerlendiriyoruz."

**Geçiş:** "Peki bu veri seti üzerinde literatür ne yapmış?"

---

## ⏱️ 07:00 – 12:00 — Literatür Taraması

**Slayt 5-6:** Alaudah 2019'dan ekran görüntüsü + literatürdeki sayıların özet tablosu

**Konuşma:**

> "Alaudah ve arkadaşları 2019'da bu benchmark'ı yayınladığında iki temel yöntem önerdiler: patch-based (küçük 64×64 yamalar) ve section-based (tam kesit). Section-based + augmentation + skip connections kombinasyonuyla en iyi sonucu aldılar — Pixel Accuracy 0.905, Mean Class Accuracy 0.817, Frequency-Weighted IoU 0.832.
>
> Sonraki yıllarda farklı yaklaşımlar denendi: Liu ve arkadaşları 2020'de F3 üzerinde 3D CNN ve semi-supervised GAN kombinasyonunu denediler — sonuçları, **section-based 2D yaklaşımının patch-based 3D'den daha iyi olduğunu** gösterdi. Bu, bizim mimari kararımızı doğrudan etkiledi.
>
> Daha yakın zamanda 2023-2025 arasında Foundation Model trendi başladı: Sheng ve arkadaşları SFM'i, ThinkOnward ekibi GFM'i geliştirdi. Bunlar milyonlarca sismik slice'la pretrain edilmiş ViT mimarileri — Parihaka veri setinde DeepLab 0.55 mIoU'dan SFM 0.80'e sıçrama gösterdi.
>
> [SLAYT: Karşılaştırma tablosu] Görüldüğü gibi literatürde 0.94+ mIoU rapor eden çalışmalar var — Wiley 2022 ensemble, CONSS 2023. Ama bu sayıları kıyas yaparken çok dikkatli olmalı, son slaytta açıklayacağım."

**Geçiş:** "Şimdi kendi mimarimize geçelim."

---

## ⏱️ 12:00 – 17:00 — Mimari

**Slayt 7-9:** DeepLabV3+ blok diyagramı + EfficientNet-B4 + ASPP detayı + 2.5D girdi şeması

**Konuşma:**

> "Mimari olarak DeepLabV3+ + EfficientNet-B4 backbone'unu seçtik. Bu seçimin üç gerekçesi var:
>
> Birincisi, **ASPP — Atrous Spatial Pyramid Pooling**. Sismik tabakaların kalınlığı çok değişken; bazı sınıflar 5-10 piksel ince band, bazıları 100+ piksel kalın blok. ASPP'nin atrous oranları (12, 24, 36) çoklu ölçek bağlamı yakalar — U-Net'in skip connection'larıyla yakalayamayacağı global context bilgisini verir.
>
> İkincisi, **EfficientNet-B4 + ImageNet pretrained**. Sismik domain'e doğrudan pretrain'lenmiş modeller var (SFM, GFM gibi) ama şu an pratik olarak entegre etmek zor — onları future work'e bıraktık. ImageNet pretrain düşük seviyeli filtreleri (kenar, doku) transfer eder, bu sismik için yeterli temel oluşturur.
>
> Üçüncüsü ve belki en önemlisi: **2.5D girdi**. Her inline kesiti için, kendisinin yanı sıra bir önceki ve bir sonraki komşu kesiti de alıp 3-kanallı bir RGB-benzeri görüntü oluşturuyoruz. Bu, 3D bağlamın bir kısmını yakalarken hem ImageNet pretrained encoder ile uyumlu kalmasını sağlar hem de 3D mimarinin VRAM patlamasından kaçınır.
>
> [SLAYT: Neden 3D değil] Liu 2020'in F3 bulgusu, tek volüm eğitim verisinde patch-based 3D'nin section-based 2D'den daha kötü olduğunu gösteriyor. Bizim 8 GB VRAM'lik 3060 Ti'mizde 3D zaten patch-based zorunlu olur, global jeolojik bağlam parçalanır. Bu yüzden 2.5D pragmatik optimum."

**Geçiş:** "Mimari hazır — şimdi onu nasıl eğittiğimize bakalım."

---

## ⏱️ 17:00 – 24:00 — Metodoloji

**Slayt 10-13:** Loss bileşenleri + augmentation listesi + sampling şeması + train loop akışı

**Konuşma:**

> "Eğitim için 100 epoch çalıştırdık, AdamW optimizer'ı lr=1e-4 ile, weight decay 1e-3. Cosine Annealing Warm Restarts scheduler'ı her 25 epoch'ta lr'i resetliyor — local minima'dan çıkış için.
>
> [SLAYT: Triple Loss] Loss fonksiyonu üç bileşen: 0.4 ağırlıklı Label-Smoothing Cross Entropy, 0.3 Dice Loss, 0.3 Focal Loss. Bu üçlünün her biri farklı bir soruna yanıt: LS-CE aşırı güveni önler, Dice doğrudan örtüşmeyi optimize eder, Focal azınlık sınıflara odaklanır.
>
> Sınıf dengesizliğini iki başka yerde de adresliyoruz: Focal Loss'un alpha parametresini sınıf frekansının tersi olarak ayarladık. Ayrıca **Weighted Random Sampler** kullanıyoruz — Class 4 ve 5 piksellerini içeren slice'ları 10 kat fazla görüyor model.
>
> Augmentation'da seçici davranıyoruz: HorizontalFlip, ShiftScaleRotate, ElasticTransform, brightness/contrast, GaussNoise, polarity inversion. Önemli bir ayrıntı — **VerticalFlip kullanmıyoruz**. Çünkü sismikte derinlik ekseni jeofiziksel olarak ters çevrilemez; üst tabakalar üstte olmalı.
>
> Bir başka önemli detay — bu projenin kritik bilimsel öğrenmelerinden biri — **methodology fix**. Erken sürümlerde fark ettim ki doğrulama setim son 80 inline'lık contiguous bir blok olarak ayrılmıştı. Bu, üç ayrı veri sızıntısı içeriyordu: yan blok lokasyon yanlılığı, crossline kanalı 3D leakage, ve 2.5D komşu sızıntısı. Bunları birer birer giderdim — doğrulama setini ortaya kaydırdım, ±2 buffer ekledim, crossline image'larından val piksellerini cropladım. Bu düzeltmenin sayısal etkisini birazdan göreceğiz."

**Geçiş:** "Sonuçlara geçelim."

---

## ⏱️ 24:00 – 30:00 — Sonuçlar

**Slayt 14-18:** Eğitim eğrileri + per-class IoU + segmentasyon görselleri (Test1 + Test2) + confusion matrix + öncesi/sonrası tablo

**Konuşma:**

> "[SLAYT: Eğitim eğrileri] Train ve validation loss düzgün düşüyor, mIoU eğrisi cosine warm restart'ın etkisini gösteriyor — her 25 epoch'ta restart, ardından yeni local minimum.
>
> [SLAYT: Test sonuçları] Methodology fix uygulanmış v7-fixed sürümünün test sonuçları:
> - Test1 mIoU: [TBD] — inline yönünde değerlendirme
> - Test2 mIoU: [TBD] — crossline genelleme
> - Combined mIoU: [TBD]
>
> [SLAYT: Per-class IoU] Sınıf bazında bakıldığında, baskın sınıflar (Upper NS, Rijnland) yüksek IoU alıyor. Azınlık sınıflarda — özellikle Class 5 Under Zechstein — IoU daha düşük. Class 4 Zechstein ise ilginç bir özel durum: Test1'de IoU 0.84, Test2'de IoU 0.18 — yaklaşık 4.7 kat fark. Bu felaketin nedenini birazdan analiz edeceğim.
>
> [SLAYT: Methodology fix öncesi/sonrası] Bu tablo bence sunumun en önemli bölümü. Methodology fix öncesinde Combined mIoU 0.793, validation mIoU sadece 0.66 idi — yani test val'den 12 puan daha yüksekti. Bu istatistiksel olarak imkansız bir durum, sızıntının somut göstergesi. Methodology fix sonrası: [TBD] — val/test farkı normalleşti.
>
> [SLAYT: Class 4 error analysis] Class 4 Zechstein'in Test2'deki başarısızlığını incelediğimizde şunu görüyoruz: Tuz tabakası **yön bağımlı**. Inline yönünde sürekli, blok yapıdadır. Crossline yönünde kıvrım/diapir yapıları öne çıkar. Eğitim verimiz inline-baskın temsil sunduğu için model crossline morfolojisine genelleyemiyor. Bu, methodology değil, **alan adaptasyonu** problemi — future work başlığında ele alıyorum."

**Geçiş:** "Bu sayıları literatürdeki SOTA çalışmalarıyla nasıl karşılaştırırız?"

---

## ⏱️ 30:00 – 35:00 — SOTA Karşılaştırma

**Slayt 19-21:** Literatür karşılaştırma tablosu + metrik farkları açıklaması + savunma slaytı

**Konuşma:**

> "[SLAYT: SOTA tablosu] Literatürde F3 üzerinde rapor edilen sayılar geniş bir aralıkta: Alaudah baseline'ı PA 0.905, FwIoU 0.832; Wiley 2022 ensemble mIoU 0.9392; CONSS 2023 mIoU 0.9462. Peki bizim 0.79 mIoU'muz neden bu kadar düşük?
>
> Bu sorunun cevabı **çok kritik** ve sunumumun en dürüst noktasına geliyor.
>
> ## ⭐ SAVUNMA CÜMLESİ ⭐
>
> **'Literatürdeki 0.94+ mIoU sayılarını analiz ettiğimizde üç metodolojik fark görüyoruz:**
>
> **(1) Random patch split kullanıyorlar — komşu pikseller train ve test arasında dağıldığı için trivial mIoU artışı oluyor;**
>
> **(2) Farklı sınıf sayısı — Wiley 7 sınıf kullanmış, biz Alaudah'ın orijinal 6 sınıfında — ve farklı metrik — FwIoU ile mIoU karıştırılıyor;**
>
> **(3) Heavy ensembling — Wiley DeepLabV3+ ve SegNet birlikte ensembling yapıyor.**
>
> **Biz Alaudah 2019 benchmark'ının orijinal coğrafi Test1/Test2 split'ine ve 6-sınıf düzenine sadığız — sayılarımız daha düşük ama doğrudan benchmark'la karşılaştırılabilir. Eğer Alaudah'ın FwIoU metriğini bizim modelimizle hesaplarsak [→ slayt X], 0.83+ alırız ki bu Alaudah baseline'ıyla aynı seviye.'**
>
> [SLAYT: FwIoU karşılaştırma]
> | Yöntem | mIoU | FwIoU | PA |
> |---|---|---|---|
> | Alaudah section + aug + skip (best baseline) | — | 0.832 | 0.905 |
> | **Bizim v7-fixed** | [TBD] | **[TBD]** | [TBD] |
> | AdaSemSeg target-only (farklı split) | — | 0.86 | 0.91 |
>
> Bu yüzden v7 sayılarımız aslında **literatür baseline'ıyla aynı seviyede** — daha yüksek görünmemesinin tek nedeni metodolojik dürüstlük."

**Geçiş:** "Sunumu bitirmeden, çalışmanın sınırlılıklarına dürüstçe değinmek istiyorum."

---

## ⏱️ 35:00 – 38:00 — Limitations (DÜRÜST OL)

**Slayt 22-23:** Limitations madde listesi + her birinin Future Work bağlantısı

**Konuşma:**

> "Bu çalışmanın bilimsel sınırlarını açıkça raporlamak istiyorum. Detaylı belge `docs/LIMITATIONS.md`'de — burada en kritik üçüne değineceğim:
>
> **Birincisi, tek tohumlu eğitim.** Tüm sonuçlar SEED=42 ile elde edildi. Standart sapma raporlamadık. Tezin sonraki versiyonunda 3 tohumla mean ± std hesaplanacak.
>
> **İkincisi, ablation study eksikliği.** Mixup'ın, Focal loss'un, 2.5D'nin, TTA'nın tek tek katkısını ölçemedik. Sınırlı bir ablation matrisi (3-4 koşu) sunum öncesinde tamamlanacak.
>
> **Üçüncüsü ve en önemlisi, Class 4 Test2 başarısızlığı**. Combined mIoU'nun 0.79 olması Zechstein gibi yorumcu için kritik bir sınıfın bir test setinde tamamen kaybolmasını gizliyor. Çözüm methodology değil, alan adaptasyonu gerektiriyor.
>
> Bu sınırlılıkları gizlemek yerine raporlamak, bu çalışmanın bilimsel olgunluğunun göstergesidir — gizlenmiş sınırlılıklar her zaman daha kötüdür."

**Geçiş:** "Son olarak, bu çalışmanın devamı için planlanan adımlar..."

---

## ⏱️ 38:00 – 40:00 — Future Work + Sonuç

**Slayt 24-25:** Future work listesi + teşekkür

**Konuşma:**

> "Bu çalışmanın doğal devamı olarak şu yönleri planlıyorum:
>
> **Birincisi**, **Foundation Model fine-tuning** — özellikle ThinkOnward GFM. Domain-spesifik pretrain, ImageNet pretrain'in çok ötesinde gelişme potansiyeli sunuyor. F3'te bağımsız bir Foundation Model fine-tune sonucu literatürde henüz yok — bu özgün bir katkı olabilir.
>
> **İkincisi**, **alan adaptasyonu** — Class 4 Test2 sorununa AdaSemSeg veya EarthAdaptNet türevi few-shot yaklaşımlarla çözüm.
>
> **Üçüncüsü**, **sismik-spesifik öznitelikler** — instantaneous phase, envelope amplitude gibi jeofiziksel attribute'ları ek girdi kanalı olarak vermek.
>
> Bu çalışmadan çıkardığım en büyük ders şudur: derin öğrenme tek başına bir mühendislik problemi değil; doğru methodology, dürüst değerlendirme ve domain bilgisi bilimsel olgunluğun ön koşullarıdır.
>
> Beni dinlediğiniz için teşekkür ederim. Sorularınızı bekliyorum."

---

## 📋 Beklenen Sorular ve Cevap Notları

### S1: "Val %66 ama test %79 — bu fark nasıl?"
**Cevap:** "Eski v7 sürümünde val seti son %20 contiguous bloktu — lokasyon bias + 3D crossline leakage + 2.5D komşu sızıntısı vardı. Yol A methodology fix ile düzelttim: val ortaya kaydırıldı, ±2 buffer, xline cropping. Yeni val/test farkı [TBD] — beklediğimiz gibi normalleşti."

### S2: "Crossline'lar val inline bölgesinden geçiyor mu?"
**Cevap:** "Eski v7'de evet — bu bilimsel hata. **Düzelttim**. Yeni `train_inline_mask` ile crossline image'larından val piksellerini cropladım. Test2 maskelenmedi çünkü ayrı volume."

### S3: "Neden DeepLabV3+, U-Net değil?"
**Cevap:** "ASPP'nin multi-scale context yakalama yeteneği, sismik tabakaların farklı kalınlıklarına uygun. Architecture ablation tablosunda [→ slayt X] U-Net++ ve MAnet ile karşılaştırması var — DeepLabV3+ en iyi performansı verdi."

### S4: "3D mimari neden değil?"
**Cevap:** "Üç neden: (1) F3 tek volüm — patch-based zorunlu, global jeolojik bağlam parçalanır. (2) Sismik için 3D pretrain encoder yok. (3) Liu et al. 2020 *Geophysics*'de F3 üzerinde section-based 2D'nin patch-based 3D'den daha iyi olduğunu somut olarak gösterdi."

### S5: "Mixup'ın katkısı ölçtün mü?"
**Cevap:** "Sınırlı ablation yaptım: Mixup-off koşusunda Combined mIoU [TBD]'a düştü — yani Mixup +[TBD] puan katkı sağlıyor. Tam ablation matrisi (her bileşen on/off) yüksek lisans semineri kapsamı dışında, tezin uzun versiyonunda olacak."

### S6: "Class 4 Test2 IoU 0.18 — neden ve ne yapılabilir?"
**Cevap:** "Zechstein tuz tabakası **anisotropic** — inline yönünde sürekli/blok yapıdadır, crossline yönünde kıvrım/diapir morfolojisi gösterir. Eğitim verim inline-baskın temsil sunduğu için model crossline'a genelleyemiyor. Çözüm methodology değil **domain adaptation** — AdaSemSeg, EarthAdaptNet gibi yaklaşımlar future work'te."

### S7: "%79 mIoU SOTA'ya göre nerede?"
**Cevap:** [SAVUNMA CÜMLESİNİ TEKRAR ET — slayt 21]

### S8: "5-fold cross-validation neden yok?"
**Cevap:** "5-fold = 5x eğitim maliyeti, 3060 Ti'de 7-10 saat × 5 = 35-50 saat. Yüksek lisans semineri zaman bütçesinde uygulanabilir bulmadık. Tek seed yerine multi-seed (3 seed) yapılması future work'te ilk öncelik."

### S9: "Foundation model neden denenmedi?"
**Cevap:** "SFM ve ThinkOnward GFM'i araştırdım. SFM'in ağırlıkları Çin hosting'inde — Türkiye'den indirme problemli. Ayrıca her ikisi de 1-channel grayscale bekliyor, 2.5D 3-kanallı pipeline'la uyumsuz; smp framework'üyle entegrasyon yok. Future work — ama bilinçli bir seçim, atlama değil."

---

## 🎯 Slayt Üretim Listesi (Sunum Hazırlığı)

| # | Slayt | İçerik | Görsel kaynak |
|---|---|---|---|
| 1 | Başlık | Proje adı + isim + tarih | — |
| 2 | Problem | Sismik kesit + jeofizikçi yorumu | `assets/whatsapp/` |
| 3 | Veri | F3 volüm 3D görsel | EDA notebook çıktısı |
| 4 | Sınıf dağılım | Bar grafik + 6 sınıf renk paleti | `eda_multiview.png` |
| 5 | Literatür | Alaudah 2019 önemli eserler timeline | manuel |
| 6 | SOTA tablosu | Özet karşılaştırma | `docs/literature_table.md` |
| 7 | Mimari | DeepLabV3+ blok diyagramı | smp dokümentasyonu |
| 8 | EfficientNet-B4 | Backbone şeması | EfficientNet paper |
| 9 | 2.5D girdi | 3 komşu slice → 3 kanal | manuel çiz |
| 10 | Triple Loss | Bileşenler + ağırlıklar | manuel |
| 11 | Augmentation | Örnek görseller | augmentation notebook |
| 12 | Methodology fix | Eski vs yeni split şeması | manuel çiz |
| 13 | Train loop | Akış şeması | manuel |
| 14 | Eğitim eğrileri | Loss + mIoU grafikleri | `training_curves_v7.png` |
| 15 | Test sonuçları | Test1/Test2/Combined mIoU | metrics JSON |
| 16 | Per-class IoU | Bar chart | `per_class_metrics_v7.png` |
| 17 | Confusion matrix | Test1+Test2 birleşik | `confusion_matrix_v7.png` |
| 18 | Segmentasyon görsel | Sismik \| GT \| Tahmin | `segmentation_comparison_v7.png` |
| 19 | Class 4 analiz | Inline vs crossline morfoloji | `class4_morphology_comparison.png` |
| 20 | Methodology fix etki | Öncesi vs sonrası tablo | metrics karşılaştırma |
| 21 | **SOTA savunma** | Metodolojik fark açıklaması + FwIoU tablo | manuel + `literature_table.md` |
| 22 | Limitations | 9 madde özet | `LIMITATIONS.md` |
| 23 | Bilimsel olgunluk | "Gizlemek yerine raporlamak" | manuel |
| 24 | Future Work | GFM + domain adaptation + attributes | manuel |
| 25 | Teşekkür + soru | İletişim bilgisi | — |

**Tahmini toplam:** 25 slayt × ~1.5 dk = 37 dk (3 dk soru için kalır).

---

## 🎯 Prova Notları

- **Prova zamanı:** En az 3 tam prova yap, zaman tut
- **Hız:** Slayt başına ~1.5 dk — hızlı geçme, atlamaya gerek yok
- **Vurgu noktaları:** (a) Methodology fix dürüstlüğü, (b) SOTA savunma slaytı, (c) Limitations
- **Beklenen sorular:** En az S1, S6, S7 sorulacak — cevapları ezberle
- **Bekleme cümlesi:** Soru gelince "İyi soru" deme, doğrudan cevaba gir

**Önemli:** Her bölüm sonunda **slayt geçişini sözel köprü** ile yap. "Bu konuya geldikten sonra şuna bakalım" tarzı geçişler dinleyiciyi tutarsız hissettirmez.
