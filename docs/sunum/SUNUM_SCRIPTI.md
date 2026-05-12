# Sunum Scripti — DeepLabV3+ ile Sismik Fasiyes Segmentasyonu

> **40 dakikalık seminer için konuşma metni.** Slayt değil, prova ederken yüksek sesle okuyacağın script. Her bölümün başında **slayt önerisi**, ortada **konuşma metni**, sonunda **geçiş cümlesi** var.
>
> **Hedef:** 15 Haziran 2026 final seminer (Bilgisayarda Görme dersi, yüksek lisans)
> **Sunum dili:** Türkçe
> **Hocaya ton:** Akademik dürüstlük + teknik yetkinlik. Övünme yok, "limitations farkındayım" var.
>
> **Ana sayı (her yerde tutarlı):** Combined mIoU = **0.791 ± 0.006** (v9 multi-seed ensemble, 3-seed softmax averaging).
> **Kanonik sayılar referansı:** [`SAYILAR_KANONIK.md`](SAYILAR_KANONIK.md) — sunum sırasında elinin altında tut.

---

## ⏱️ 00:00 – 03:00 — Açılış ve Problem Tanımı

**Slayt 1-2:** Başlık + sismik kesit görseli + jeofizikçi manuel yorumlama görseli

**Konuşma:**

> "İyi günler. Bugün size yüksek lisans projemde geliştirdiğim derin öğrenme tabanlı sismik fasiyes segmentasyon yaklaşımını sunacağım.
>
> Sismik fasiyes sınıflandırması, yeraltı jeolojik yapılarının sismik dalga yansıma örüntülerine göre kategorilere ayrılması işlemidir. Petrol ve doğal gaz endüstrisinde rezervuar karakterizasyonu için kritiktir. Şu an bu yorum hâlâ jeofizikçiler tarafından **elle yapılır** — bir uzman bir 3D sismik bloğu yorumlamak için günler harcar, üstelik iki farklı uzman aynı kesit için farklı yorumlar verebilir.
>
> Bu çalışmanın amacı, derin öğrenme ile bu yorumlama sürecini **piksel bazında otomatikleştirmek**. Hedef veri seti: Hollanda F3 bloğu, Alaudah 2019 benchmark.
>
> Bu sunumun üç ana mesajı olacak: (1) Modern bir 2.5D DeepLabV3+ baseline'ı kurduk, (2) Erken sürümde fark ettiğim veri sızıntısı hatalarını düzeltip dürüstçe rapor ettim, (3) Multi-seed ensemble ile **tek-seed raporlamanın istatistiksel risklerini** somut bir bulgu olarak gösterdim."

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

## ⏱️ 07:00 – 11:00 — Literatür Taraması

**Slayt 5-6:** Alaudah 2019'dan ekran görüntüsü + literatürdeki sayıların özet tablosu

**Konuşma:**

> "Alaudah ve arkadaşları 2019'da bu benchmark'ı yayınladığında iki temel yöntem önerdiler: patch-based (küçük 64×64 yamalar) ve section-based (tam kesit). Section-based + augmentation + skip connections kombinasyonuyla en iyi sonucu aldılar — Pixel Accuracy 0.905, Mean Class Accuracy 0.817, Frequency-Weighted IoU 0.832.
>
> Sonraki yıllarda farklı yaklaşımlar denendi: Liu ve arkadaşları 2020'de F3 üzerinde 3D CNN ve semi-supervised GAN kombinasyonunu denediler — sonuçları, **section-based 2D yaklaşımının patch-based 3D'den daha iyi olduğunu** gösterdi. Bu, bizim mimari kararımızı doğrudan etkiledi.
>
> Daha yakın zamanda 2023-2025 arasında Foundation Model trendi başladı: Sheng ve arkadaşları SFM'i, ThinkOnward ekibi GFM'i geliştirdi. Bunlar milyonlarca sismik slice'la pretrain edilmiş ViT mimarileri — Parihaka veri setinde DeepLab 0.55 mIoU'dan SFM 0.80'e sıçrama gösterdi. Biz de bu sunumda SFM ile yaptığımız mini-deneylere yer vereceğiz.
>
> [SLAYT: Karşılaştırma tablosu] Görüldüğü gibi literatürde 0.94+ mIoU rapor eden çalışmalar var — Wiley 2022 ensemble, CONSS 2023. Ama bu sayıları kıyas yaparken çok dikkatli olmalı, sunum sonunda açıklayacağım."

**Geçiş:** "Şimdi kendi mimarimize geçelim."

---

## ⏱️ 11:00 – 15:00 — Mimari

**Slayt 7-9:** DeepLabV3+ blok diyagramı + EfficientNet-B4 + ASPP detayı + 2.5D girdi şeması

**Konuşma:**

> "Mimari olarak DeepLabV3+ + EfficientNet-B4 backbone'unu seçtik. Bu seçimin üç gerekçesi var:
>
> Birincisi, **ASPP — Atrous Spatial Pyramid Pooling**. Sismik tabakaların kalınlığı çok değişken; bazı sınıflar 5-10 piksel ince band, bazıları 100+ piksel kalın blok. ASPP'nin atrous oranları (12, 24, 36) çoklu ölçek bağlamı yakalar — U-Net'in skip connection'larıyla yakalayamayacağı global context bilgisini verir.
>
> İkincisi, **EfficientNet-B4 + ImageNet pretrained**. Sismik domain'e doğrudan pretrain'lenmiş modeller var (SFM, GFM gibi) — bunları sunum sonunda ayrı bir mini-bölümde tartışacağım. ImageNet pretrain düşük seviyeli filtreleri (kenar, doku) transfer eder; bu sismik için yeterli bir temel oluşturur.
>
> Üçüncüsü, **2.5D girdi**. Her inline kesiti için, kendisinin yanı sıra **iki önceki ve iki sonraki** komşu kesiti de alıp **5-kanallı** bir görüntü oluşturuyoruz. Bu, 3D bağlamın bir kısmını yakalarken hem ImageNet pretrained encoder ile uyumlu kalmasını sağlar hem de 3D mimarinin VRAM patlamasından kaçınır. Erken sürümlerde 3-kanallı (±1 komşu) idi; Class 4'ün Test2 zayıflığını ele almak için 5 kanala (±2 komşu) genişlettik.
>
> [SLAYT: Neden 3D değil] Liu 2020'in F3 bulgusu, tek volüm eğitim verisinde patch-based 3D'nin section-based 2D'den daha kötü olduğunu gösteriyor. Bizim 8 GB VRAM'lik 3060 Ti'mizde 3D zaten patch-based zorunlu olur, global jeolojik bağlam parçalanır. Bu yüzden 2.5D pragmatik optimum."

**Geçiş:** "Mimari hazır — şimdi onu nasıl eğittiğimize bakalım."

---

## ⏱️ 15:00 – 21:00 — Metodoloji

**Slayt 10-13:** Loss bileşenleri + augmentation listesi + sampling şeması + methodology fix şeması

**Konuşma:**

> "Eğitim için 100 epoch çalıştırdık, AdamW optimizer'ı lr=1e-4 ile, weight decay 1e-3. Cosine Annealing Warm Restarts scheduler'ı her 25 epoch'ta lr'i resetliyor — local minima'dan çıkış için.
>
> [SLAYT: QuadrupleLoss] Loss fonksiyonu **dört bileşen**: 0.30 ağırlıklı Label-Smoothing Cross Entropy, 0.25 Dice Loss, 0.25 Focal Loss ve 0.20 **Lovász-Softmax**. İlk üç klasik kombinasyon — LS-CE aşırı güveni önler, Dice doğrudan örtüşmeyi optimize eder, Focal azınlık sınıflara odaklanır. Lovász-Softmax ise daha az tanıdık olabilir: Berman ve arkadaşları 2018'de önerdi, **IoU metriğinin türevlenebilir bir surrogate'i**. Yani diğer loss'lar dolaylı yoldan IoU'yu optimize ederken Lovász doğrudan optimize ediyor. Class 4 odaklı pakette eklendi ve sayısal olarak +1 puan civarında katkı sağladı.
>
> Sınıf dengesizliğini iki başka yerde de adresliyoruz: Focal Loss'un alpha parametresini sınıf frekansının tersi olarak ayarladık. Ayrıca **Weighted Random Sampler** kullanıyoruz — Class 4 ve 5 piksellerini içeren slice'ları 10 kat fazla görüyor model.
>
> Augmentation'da seçici davranıyoruz: HorizontalFlip, ShiftScaleRotate, ElasticTransform, brightness/contrast, GaussNoise, polarity inversion. Class 4'ün kıvrımlı/diapir morfolojisi için **xline-aware augmentation** ekledik — crossline modunda daha agresif elastic ve grid distortion. Önemli bir ayrıntı — **VerticalFlip kullanmıyoruz**. Çünkü sismikte derinlik ekseni jeofiziksel olarak ters çevrilemez; üst tabakalar üstte olmalı.
>
> [SLAYT: Methodology fix şeması] Şimdi bu projenin **en kritik bilimsel öğrenmesine** geleyim. Erken sürümlerde fark ettim ki kurulumumda **üç ayrı veri sızıntısı** vardı:
>
> **Birinci sızıntı: contiguous validation bias.** Doğrulama setim son %20 inline'lık bir bloktu — inline 320–400 arası. F3'te Zechstein tuz yapılarının dağılımı inline boyunca homojen değildir, dolayısıyla bu blok eğitim bölgesinden sistematik biçimde farklı geometriye sahip olabilir. Bu yanlılığın somut göstergesi: v7-broken'da val mIoU 0.66 idi, ama test mIoU 0.79 — test, val'den **12 puan daha yüksekti**. Bu istatistiksel olarak imkansız bir durum.
>
> **İkinci sızıntı: 3D crossline leakage.** v7-broken'da 601 crossline kesitinin tamamı eğitime dahil ediliyordu. Her crossline 2D görüntüsü 401 inline × 255 derinlik şeklinde ve **her görüntü doğrulama inline'larına ait pikselleri içeriyor**. Yani crossline kanalından val piksellerini eğitime gizli yoldan sokmuş oluyordum.
>
> **Üçüncü sızıntı: 2.5D komşu sızıntısı.** Val sınırına oturan inline'larda 2.5D komşu kanalları eğitim tarafından geliyordu. Val örneklerinin yaklaşık üçte ikisi 'yarı eğitim görmüş' oluyordu.
>
> [SLAYT: Yol A fix] Bu üç sızıntıyı şu şekilde düzelttim — buna 'Yol A methodology fix' diyorum: Validation bölgesini **ortaya kaydırdım** — inline [160, 240) arası 80 slice. Train bölgesi iki ayrı blok hâline geldi: [0, 158) ve [242, 401). Bu, val'i train ile **çevreliyor**, lokasyon bias'ını kırıyor. Validation sınırının ±2 inline'ı buffer olarak bırakıldı — 2.5D komşu sızıntısı önlendi. Crossline image'larından da val piksellerini cropladım — image'lar artık val içermiyor.
>
> Bu düzeltmenin etkisini birazdan, sonuçlar slaytında somut sayılarla göstereceğim."

**Geçiş:** "Sonuçlara geçelim."

---

## ⏱️ 21:00 – 28:00 — Sonuçlar

**Slayt 14-19:** Eğitim eğrileri + per-class IoU + segmentasyon görselleri (Test1 + Test2) + confusion matrix + methodology fix öncesi/sonrası tablo

**Konuşma:**

> "[SLAYT: Eğitim eğrileri] Train ve validation loss düzgün düşüyor, mIoU eğrisi cosine warm restart'ın etkisini gösteriyor — her 25 epoch'ta restart, ardından yeni local minimum.
>
> [SLAYT: Methodology fix öncesi/sonrası] Sunumun **en önemli slaydı** bence bu.
>
> | Sürüm | val mIoU | Test1 mIoU | Test2 mIoU | Combined mIoU | Yorumlama |
> |---|---:|---:|---:|---:|---|
> | v7-broken | 0.66 | 0.79 | 0.70 | 0.79 | val << test, **leakage anomalisi** |
> | v7-fixed | 0.84 | 0.76 | 0.67 | 0.77 | val ≥ test, normal pattern — gerçek baseline |
> | **v9 ensemble** | **0.81** | **0.80** | 0.68 | **0.791 ± 0.006** | mimari iyileştirme + multi-seed |
>
> v7-broken sürümünde Combined mIoU 0.79 görünüyordu ama bu **yapay olarak şişirilmiş bir sayıydı** — sızıntıdan besleniyordu. Methodology fix sonrası Combined 0.77'ye düştü. **Bu düşüş başarı göstergesidir**: yapay sayı yerine gerçek genelleme performansı ortaya çıktı. Sonrasında Class 4 odaklı mimari iyileştirmeler (5-kanal 2.5D, Lovász loss, xline-aware augmentation, 384×384 çözünürlük) ve 3-seed ensemble ile **gerçek bir +2.4 puanlık ilerleme** sağlandı — Combined 0.77'den 0.791'e.
>
> [SLAYT: v9 ensemble tam metrikler] Ana sonuçlar:
> - Test1 mIoU: **80.43%** — inline yönünde değerlendirme
> - Test2 mIoU: **67.71%** — crossline genelleme (kritik test)
> - Combined mIoU: **79.10% ± 0.6%** (3-seed std)
> - Pixel Accuracy: **94.02%**, Mean Class Accuracy: **87.34%**
>
> Önemli not: **standart sapma raporladık** — 3 bağımsız seed (42, 43, 44) ile mean ± std. Bu, derin öğrenme deneylerinde tek-koşu raporlamanın istatistiksel zayıflığını adresliyor.
>
> [SLAYT: Per-class IoU bar chart] Sınıf bazında bakıldığında, baskın sınıflar (Upper NS 95.1%, Rijnland 94.1%) yüksek IoU alıyor. Azınlık sınıflarda — özellikle Class 5 Under Zechstein 60.9% ve Class 3 Scruff 62.9% — IoU daha düşük. Class 4 Zechstein ise ilginç bir özel durum: Test1'de IoU 84.3%, Test2'de IoU sadece **18.3%** — 4.6 kat fark. Bu felaketin nedenini birazdan analiz edeceğim.
>
> [SLAYT: Class 4 hata analizi] Class 4 Zechstein'in Test2'deki başarısızlığını detaylı incelediğimizde net bir tablo çıkıyor: Class 4 GT piksellerinin sadece **%26'sı doğru** Zechstein olarak tahmin ediliyor; geri kalanın büyük kısmı **Under Zechstein** (S5) olarak yanlış sınıflandırılıyor — yani komşu tabaka olarak. Bu, tesadüfi bir hata değil: tuz tabakası **yön bağımlı (anisotropic)** — inline yönünde sürekli/blok yapıdadır, crossline yönünde diapir/kıvrım morfolojisi gösterir. Eğitim verimiz inline-baskın temsil sunduğu için model crossline morfolojisine genelleyemiyor. Bu, methodology değil, **alan adaptasyonu** problemi — future work başlığında ele alıyorum.
>
> [SLAYT: Segmentasyon karşılaştırma görselleri] Sismik girdi — ground truth — model tahmini üçlüsü. Test1'de tahminler GT ile neredeyse örtüşüyor. Test2'de baskın sınıflarda örtüşme iyi; Zechstein bölgesinde model boyut/sınır hatası yapıyor ama tam kaçırmıyor."

**Geçiş:** "Bu sonuçlardan çıkan kritik bir akademik bulguya geçmek istiyorum."

---

## ⏱️ 28:00 – 31:00 — Single-Seed Bias Bulgusu (KRİTİK)

**Slayt 20-21:** Class 4 Test2 IoU seed-bazlı dağılım grafiği + tek-seed vs ensemble karşılaştırma tablosu

**Konuşma:**

> "Sunumda paylaşmak istediğim **akademik açıdan en değerli bulgu** bu. Multi-seed deneyimimiz sırasında beklenmedik bir gözlem yaptık.
>
> [SLAYT: Class 4 Test2 seed dağılımı] v9 ana sürümümüzü SEED=42 ile eğittiğimizde Class 4 Test2 IoU **0.230** olarak çıkmıştı. Bu sayı v7-fixed'in 0.118'inden göreli olarak %96 daha iyiydi — sevindirici bir gelişme. Sunuma 'Class 4 sorununu yarı yarıya çözdük' diye gelmeyi planlıyorduk.
>
> Ancak 3-seed ensemble için SEED 43 ve SEED 44 ile aynı mimariyi yeniden eğittiğimizde **sürpriz** ortaya çıktı:
>
> | SEED | Class 4 Test2 IoU |
> |---|---:|
> | 42 (ilk raporlanan) | **0.230** ← outlier |
> | 43 | 0.156 |
> | 44 | 0.163 |
> | **Mean ± Std** | **0.183 ± 0.04** |
>
> Yani SEED=42'nin 0.230 sonucu **istatistiksel bir outlier'dı** — şanslı bir başlangıç. Gerçek beklenen değer 0.183 ± 0.04.
>
> Bu bize üç önemli ders veriyor:
>
> **Birincisi**, modelin Class 4 Test2 üzerindeki gerçek performansı **0.230 değil 0.183**. Tezde bu daha düşük sayıyı dürüstçe raporluyoruz.
>
> **İkincisi**, eğer tek-seed sonuçla yetinseydik, literatüre **şanslı çıkış**ı 'mimari katkı' olarak rapor edecektik — bu bilim değil, gürültü. Multi-seed analiz bu yanılgıyı önledi.
>
> **Üçüncüsü**, ve bu daha geneldir: segmentation literatüründe pek çok yayın hâlâ tek-seed sonuç raporluyor. Bu bulgumuz — küçük bir ölçekte de olsa — bu pratiğin somut riskini gösteriyor.
>
> Bu nedenle bu sunum boyunca ana sayımız **Combined mIoU 0.791 ± 0.006** — varyans dahil, multi-seed mean — olarak sunuluyor. Tek-seed kısa-yolundan kaçınıyoruz."

**Geçiş:** "İlgili bir başka mini-deneye, Foundation Model'lara değineyim."

---

## ⏱️ 31:00 – 33:30 — SFM Mini-Bölüm

**Slayt 22:** SFM trade-off matrisi tablo + Class 4 Test2'de SFM v1 lider olduğu gösterilen bar grafik

**Konuşma:**

> "Sismik domain'e özel pretrain'lenmiş ViT modeller son 2 yılın trendi. Bunlardan en bilineni SFM — Sheng ve arkadaşları 2024, 192 sismik survey'den 2.3 milyon slice ile MAE pretrain etmiş bir ViT-B/16 backbone.
>
> Ben de F3 üzerinde SFM fine-tune deneyleri yaptım — literatürde F3 facies için SFM sayıları yoktu. Sonuçlar bir **trade-off matrisi** olarak özetlenebilir:
>
> | Metrik | v9 ensemble (ana) | SFM v1 | SFM v2 | Lider |
> |---|---:|---:|---:|---|
> | Combined mIoU | **0.791** | 0.728 | 0.768 | v9 |
> | Test1 mIoU | 0.804 | 0.721 | 0.782 | v9 |
> | Test2 mIoU | 0.677 | 0.667 | 0.659 | v9 |
> | **Class 4 Test2 IoU** | 0.183 | **0.270** | 0.20 | **SFM v1** |
> | Best val mIoU | 0.81 | 0.81 | **0.823** | SFM v2 |
>
> İki net bulgu:
>
> **Bulgu 1:** SFM ana modeli sayısal olarak **geçemedi**. ImageNet pretrained DeepLabV3+, F3'ün Alaudah split'inde hâlâ daha iyi. Bunun nedeni muhtemelen SFM'in mimarisinin (1-channel ViT) bizim 2.5D + DeepLabV3+ pipeline'ımız kadar inductive bias taşımaması.
>
> **Bulgu 2:** SFM v1, **Class 4 Test2'de en iyi sonucu** üretti — 0.270. Yani domain-spesifik pretrain, tüm metriklerde değil ama **zorlu sınıfta** belirgin avantaj sağlıyor. Bu, future work için ip ucu: SFM ve ImageNet pretrained modelleri **heterojen bir ensemble**'da birleştirmek Class 4 problemini daha iyi adresleyebilir.
>
> SFM ana model olarak seçilmedi çünkü ortalama metriklerde geride. Ama bu sonuçlar tezin Discussion bölümünde önemli — domain-pretrain trade-off'unun somut bir kanıtı."

**Geçiş:** "Şimdi tüm bu sayıları literatürle nasıl karşılaştıracağımıza bakalım."

---

## ⏱️ 33:30 – 36:00 — Literatür Karşılaştırma

**Slayt 23-24:** Literatür karşılaştırma tablosu + savunma slaydı

**Konuşma:**

> "[SLAYT: SOTA tablosu] Literatürde F3 üzerinde rapor edilen sayılar geniş bir aralıkta: Alaudah baseline'ı PA 0.905, FwIoU 0.832; Wiley 2022 ensemble mIoU 0.9392; CONSS 2023 mIoU 0.9462. Peki bizim 0.791 mIoU'muz neden bu kadar düşük?
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
> **(3) Heavy ensembling + interactive prompts — UmixClick 2025 bir kullanıcı tıklamalı yöntem.**
>
> **Biz Alaudah 2019 benchmark'ının orijinal coğrafi Test1/Test2 split'ine ve 6-sınıf düzenine sadığız. Modelimizden Alaudah ile aynı metriklerle (PA, MCA) hesapladığımızda PA 0.940, MCA 0.873 elde ediyoruz — Alaudah'ın 'best baseline'ının PA 0.905 / MCA 0.817 değerlerine yakın bantta. ÖNEMLİ KAVET: Bizim sayılarımız 384×384 resized space'te hesaplanmıştır; Alaudah orijinal çözünürlükte (701×255 / 401×255) evaluator çalıştırır. Doğrudan "geçtik" iddiası için aynı evaluator protokolünü çalıştırmak gerekir — bu original-resolution evaluator henüz yazılmadı, bekleyen iş. Şu anki dürüst ifade: **Alaudah baseline seviyesinde, methodology hataları analiz edilip düzeltilmiş, multi-seed ile varyans ölçülmüş bir baseline**. mIoU sayımız 0.791 — bu metriği Alaudah rapor etmediği için doğrudan kıyas yok.'**
>
> [SLAYT: FwIoU/PA/MCA karşılaştırma]
> | Yöntem | mIoU | PA | MCA | Eval Çözünürlüğü |
> |---|---:|---:|---:|---|
> | Alaudah section + aug + skip (best baseline) | — | 0.905 | 0.817 | Orijinal (701×255) |
> | **Bizim v9 ensemble** | **0.791 ± 0.006** | **0.940** | **0.873** | **384×384 resized** |
> | AdaSemSeg target-only (farklı split) | — | 0.91 | 0.89 | farklı |
> | UmixClick (interactive) | 0.767 | 0.935 | — | belirsiz |
>
> v9 ensemble PA ve MCA sayıları Alaudah baseline'ı seviyesinde — ancak farklı evaluator protokolü kullandığımız için bu doğrudan kıyas değildir. **Şu anki dürüst iddia: 'Alaudah baseline seviyesinde, leakage hatalarını analiz edip düzelten, multi-seed varyans raporlayan dürüst bir baseline'**."

**Geçiş:** "Sunumu bitirmeden, çalışmanın sınırlılıklarına dürüstçe değinmek istiyorum."

---

## ⏱️ 36:00 – 38:00 — Limitations (DÜRÜST OL)

**Slayt 25-26:** Limitations madde listesi + her birinin Future Work bağlantısı

**Konuşma:**

> "Bu çalışmanın bilimsel sınırlarını açıkça raporlamak istiyorum. Detaylı belge `docs/LIMITATIONS.md`'de — burada en kritik üçüne değineceğim:
>
> **Birincisi, Class 4 Test2 başarısızlığı**. Ensemble'da bile Zechstein için Test2 IoU sadece 0.183 — yorumcu için kritik bir sınıfın bir test setinde model güvenilir değil. Bu methodology değil **alan adaptasyonu** problemi.
>
> **İkincisi, original-resolution evaluator eksikliği**. Sayılarımız 384×384 resized space'te hesaplandı. Alaudah orijinal çözünürlükte değerlendiriyor. Bu olmadan 'Alaudah'ı geçtik' iddiası yapamıyoruz — sadece 'seviyesindeyiz' diyoruz. Bu evaluator yazımı bekleyen iş.
>
> **Üçüncüsü, tek-volüm eğitim verisi**. F3 üzerinde eğittik ve test ettik. Penobscot veya Parihaka gibi farklı jeolojik ortamlarda performans bilinmiyor. Çapraz-volüm transfer testi bu çalışma kapsamında değildi.
>
> Bunlara ek olarak: K-fold cross-validation yapılmadı (5x GPU maliyeti), tam ablation matrisi (2^6 = 64 kombinasyon) yapılmadı — sınırlı ablation paketi yapıldı. Bunların hepsi `docs/LIMITATIONS.md`'de detaylı yazılı.
>
> Bu sınırlılıkları gizlemek yerine raporlamak, bu çalışmanın bilimsel olgunluğunun göstergesidir — gizlenmiş sınırlılıklar her zaman daha kötüdür."

**Geçiş:** "Son olarak, bu çalışmanın devamı için planlanan adımlar..."

---

## ⏱️ 38:00 – 40:00 — Future Work + Sonuç

**Slayt 27-28:** Future work listesi + teşekkür

**Konuşma:**

> "Bu çalışmanın doğal devamı olarak şu yönleri planlıyorum:
>
> **Birincisi**, **heterojen ensemble** — v9 + SFM v1 softmax averaging. Bugünkü mini-deneyde gördük ki SFM v1 Class 4 Test2'de daha iyi. İki modeli birleştirmek Class 4 problemini muhtemelen daha iyi adresler. Düşük maliyetli, yüksek potansiyelli bir deney.
>
> **İkincisi**, **alan adaptasyonu** — Class 4 Test2 sorununa AdaSemSeg veya EarthAdaptNet türevi few-shot yaklaşımlarla çözüm. Test2'deki yön bağımlı genelleme zafiyetini doğrudan adresler.
>
> **Üçüncüsü**, **original-resolution evaluator** — Alaudah ile birebir karşılaştırılabilir sayılar üretmek için. Eksik kalan en somut bekleyen iş.
>
> **Dördüncüsü**, **çapraz-volüm transfer** — F3'te eğit, Penobscot veya Parihaka'da test et. Modelin gerçek genelleme kapasitesini ölçmek için.
>
> Bu çalışmadan çıkardığım en büyük ders şudur: derin öğrenme tek başına bir mühendislik problemi değil; **doğru methodology, dürüst değerlendirme ve istatistiksel rigor** bilimsel olgunluğun ön koşullarıdır. Single-seed bias bulgumuz bu prensibin somut bir örneğidir.
>
> Beni dinlediğiniz için teşekkür ederim. Sorularınızı bekliyorum."

---

## 📋 Beklenen Sorular ve Cevap Notları

### S1: "Val %66 ama test %79 — bu fark nasıl?"
**Cevap:** "Eski v7-broken sürümünde val seti son %20 contiguous bloktu — lokasyon bias + 3D crossline leakage + 2.5D komşu sızıntısı vardı. Yol A methodology fix ile düzelttim: val ortaya kaydırıldı, ±2 buffer, xline cropping. Yeni val/test farkı **best val 0.81 vs Combined test 0.79 — normal istatistiksel pattern** (val ≥ test, beklediğimiz gibi). Combined mIoU **0.79'dan 0.77'ye düştü** (v7-fixed), sonra mimari iyileştirme + ensemble ile **0.791'e** ulaştı. İlk düşüş beklenmedik bir başarı: yapay olarak şişirilmiş sayı yerine gerçek baseline ortaya çıktı."

### S2: "Crossline'lar val inline bölgesinden geçiyor mu?"
**Cevap:** "Eski v7-broken'da evet — bu bilimsel hata. **Düzelttim**. Yeni `train_inline_mask` ile crossline image'larından val piksellerini cropladım. Test2 maskelenmedi çünkü ayrı volume."

### S3: "Neden DeepLabV3+, U-Net değil?"
**Cevap:** "ASPP'nin multi-scale context yakalama yeteneği, sismik tabakaların farklı kalınlıklarına uygun. U-Net'in skip connection'ları lokal bağlamı yakalamada güçlü ama global context için sınırlı. v3 baseline'ımız plain U-Net'ti (mIoU 0.40), v5'te DeepLabV3+ + EfficientNet-B4'e geçişle 0.76'ya sıçradı — bu mimari kararının somut etkisi."

### S4: "3D mimari neden değil?"
**Cevap:** "Üç neden: (1) F3 tek volüm — patch-based zorunlu, global jeolojik bağlam parçalanır. (2) Sismik için 3D pretrain encoder yok. (3) Liu et al. 2020 *Geophysics*'de F3 üzerinde section-based 2D'nin patch-based 3D'den daha iyi olduğunu somut olarak gösterdi. Bizim 2.5D yaklaşımımız — 5 komşu slice'ı kanal olarak vermek — 3D bağlamın bir kısmını yakalarken ImageNet pretrained encoder ile uyumlu kalıyor."

### S5: "Mixup'ın katkısı ölçtün mü? Ablation var mı?"
**Cevap:** "Sınırlı ablation paketi yaptım — TTA on/off, Lovász on/off, 5-channel vs 3-channel ana koşular. TTA Combined mIoU'yu ~+1 puan, Lovász ~+1 puan, 5-channel Class 4 Test2'yi ~+5 puan katkı yapıyor. Mixup, label smoothing, focal alpha gibi diğer bileşenler için tam matris (2^6 kombinasyon) yüksek lisans semineri zaman bütçesinde sığmıyor — tezin uzun versiyonunda planlanıyor."

### S6: "Class 4 Test2 IoU 0.18 — neden ve ne yapılabilir?"
**Cevap:** "Zechstein tuz tabakası **anisotropic** — inline yönünde sürekli/blok yapıdadır, crossline yönünde kıvrım/diapir morfolojisi gösterir. Eğitim verim inline-baskın temsil sunduğu için model crossline'a genelleyemiyor. Bu **domain adaptation** problemi — methodology değil. AdaSemSeg, EarthAdaptNet gibi few-shot DA yaklaşımları future work'te. Ayrıca bugün gösterdiğim SFM mini-deneyinde SFM v1'in Class 4 Test2'de 0.270 ile daha iyi olduğunu gördük — heterojen ensemble (v9 + SFM) ucuz ve umut verici bir çözüm yolu."

### S7: "%79 mIoU SOTA'ya göre nerede?"
**Cevap:** "Mevcut çalışma SOTA iddiasında değil. Modern literatür (2025-2026) sismik-spesifik foundation model'lara döndü: SFM 192 sismik survey'den 2.3M slice ile pretrain ediyor, GFM 450 sentetik volüm kullanıyor. Bizim ImageNet pretrain EfficientNet-B4 yaklaşımımız bu perspektifte **baseline seviyesinde**. F3 baseline'ı (Alaudah 2019 PA 0.905 / MCA 0.817) ile benzer bantta — ama doğrudan kıyas için orijinal-çözünürlük evaluator gerekli, bu future work. Literatürdeki 0.94+ sayıların çoğu farklı split/farklı metrik (savunma cümlesi açıklandı)."

### S8: "5-fold cross-validation neden yok?"
**Cevap:** "5-fold = 5x eğitim maliyeti, 3060 Ti'de 7-10 saat × 5 = 35-50 saat. Yüksek lisans semineri zaman bütçesinde uygulanabilir bulmadım. Bunun yerine **3-seed multi-seed ensemble** yaptım — varyans ölçümü için bu da geçerli bir yaklaşım, hatta nnU-Net pratiği bu yöndedir. K-fold tezin uzun versiyonunda planlanıyor."

### S9: "Foundation model neden denenmedi?"
**Cevap (güncellenmiş):** "Denedim — sunumun **SFM Mini-Bölümünde** gösterdim. Sismik Foundation Model'i (SFM, Sheng 2024) F3'te fine-tune ettim. Sonuç: ortalama metriklerde ana modeli geçemedi (Combined mIoU 0.728/0.768 vs 0.791), ama **Class 4 Test2'de en iyi sonuç (0.270 vs bizim 0.183)** SFM v1'den geldi. Yani domain-pretrain trade-off var: tüm metriklerde değil, zorlu sınıflarda avantaj. Heterojen ensemble (v9 + SFM) future work."

### S10 (YENİ): "v9 vs v7-c4fix sayıları çok yakın (0.777 vs 0.778) — v9'a geçmenin justification'ı ne?"
**Cevap:** "Combined mIoU'da fark az, doğru. Ama v9'un asıl katkısı **Class 4 Test2 üzerinde** — v7-c4fix 0.182 vs v9 0.230 (single-seed). Yani ana model değişimi Class 4 sorununa odaklı bir karardı. Daha sonra multi-seed analizinde v9'un 0.230'unun outlier olduğunu, gerçek değerin 0.183 olduğunu gördük — bu da single-seed bias bulgumuzun yapı taşı. Ayrıca 384×384 çözünürlük + multi-scale TTA gelecekteki cross-volume testi için daha esnek bir taban."

### S11 (YENİ): "SFM Class 4'te daha iyi ise neden ana model SFM değil?"
**Cevap:** "İki gerekçe: (1) **Ortalama metriklerde geride** — Combined mIoU SFM v1 0.728, SFM v2 0.768, bizim v9 ensemble 0.791. Tez ana modelin overall performansta lider olmasını gerektiriyor. (2) **Heterojen ensemble future work** — v9 + SFM birleşimi muhtemelen her ikisinin de gücünü taşır. Şu an seçim 'sayısal en iyi' olmak zorundaydı; mantıksal en iyi 'birleşim' future work'tedir."

### S12 (YENİ): "Original-resolution evaluator olmadan PA/MCA karşılaştırması anlamlı mı?"
**Cevap:** "Tamamen anlamlı değil — bu yüzden 'geçtik' iddiası **yapmıyoruz**. Sadece **yakın bantta** olduğumuzu söylüyoruz. Resize evaluatöründe PA ve MCA sayıları aşağı yukarı eşit kalır (piksel başına metrikler), mIoU/FwIoU resize'a daha duyarlıdır. Original-resolution evaluator yazımı bekleyen iş; gerçek 'geçtik mi?' cevabı oradan gelecek. Bunu Limitations'da açıkça yazıyorum."

### S13 (YENİ): "3-seed yeterli mi? 10-seed veya 30-seed olsa nasıl olurdu?"
**Cevap:** "İdeal değil ama yüksek lisans semineri için makul. 3-seed std 0.006 — bu zaten oldukça düşük, demek ki Combined mIoU'da varyans az. Class 4 Test2 std'si 0.04 daha yüksek — bu konuda 3-seed sınırlı tahmin verir. Tezin uzun versiyonunda en az 5-seed planlanıyor. Çok-seed (30+) literatürde nnU-Net stilinde nadir görülür, çünkü ekstra 7 koşunun marjinal değeri düşüktür — gauge için 3-5 yeterli."

### S14 (YENİ): "Lovász loss + AMP FP32 wrap — bu bir engineering hack mı?"
**Cevap:** "Hack'ten ziyade 'AMP'ın bilinen kısıtı'. Lovász-Softmax sıralama-tabanlı bir loss, fp16 hassasiyeti yetmiyor, NaN üretiyor. Bunu **diğer loss'ları fp16'da bırakıp sadece Lovász'ı fp32'de hesaplayarak** çözdüm — hibrit precision. Bu PyTorch'ta standart bir teknik (autocast disable context). Gradient scaler bozulmuyor. Tez metninde nüans olarak raporlanıyor."

---

## 🎯 Slayt Üretim Listesi (Sunum Hazırlığı)

| # | Slayt | İçerik | Görsel kaynak |
|---|---|---|---|
| 1 | Başlık | Proje adı + isim + tarih | — |
| 2 | Problem | Sismik kesit + jeofizikçi yorumu | `assets/whatsapp/` |
| 3 | Veri | F3 volüm 3D görsel | EDA notebook çıktısı |
| 4 | Sınıf dağılım | Bar grafik + 6 sınıf renk paleti | `eda_multiview.png` |
| 5 | Literatür timeline | Alaudah 2019, Liu 2020, SFM 2024, GFM 2025 | manuel |
| 6 | SOTA tablosu özet | Karşılaştırılabilir/karşılaştırılamaz ayrımı | `literature_table.md` |
| 7 | DeepLabV3+ mimari | Encoder + ASPP + decoder blok diyagramı | **manuel diyagram (Task 10)** |
| 8 | EfficientNet-B4 + ASPP | Backbone + ASPP detayı | smp + EfficientNet paper |
| 9 | 2.5D girdi | 5 komşu slice → 5 kanal şeması | **manuel diyagram (Task 8)** |
| 10 | QuadrupleLoss | 4 bileşen + ağırlıklar + her bileşenin rolü | manuel |
| 11 | Augmentation | Örnek görseller + xline-aware aug açıklama | augmentation notebook |
| 12 | Methodology fix şeması | 3 sızıntı türünün görsel anlatımı | **manuel diyagram (Task 9)** |
| 13 | Yol A fix sonrası | Yeni split şeması (val ortada, buffer'lı, cropped xline) | **manuel diyagram (Task 9)** |
| 14 | Eğitim eğrileri | Loss + mIoU grafikleri (v9) | `training_curves_v9.png` |
| 15 | **Methodology fix öncesi/sonrası tablo** | v7-broken → v7-fixed → v9 ensemble | manuel tablo |
| 16 | v9 ensemble tam metrikler | Test1/Test2/Combined tablosu | `v9_ensemble_metrics.json` |
| 17 | Per-class IoU | Bar chart (ensemble combined) | `per_class_metrics_v9.png` |
| 18 | Confusion matrix | v9 ensemble Test1+Test2 birleşik | `confusion_matrix_v9.png` |
| 19 | Segmentasyon görsel | Sismik \| GT \| Tahmin | `segmentation_comparison_v9.png` |
| 20 | **Single-seed bias bulgusu** | Class 4 Test2 seed-bazlı dağılım | **manuel + JSON** |
| 21 | **Single-seed bias** kompozisyon | 0.230 outlier vs 0.183 ensemble karşılaştırma | manuel tablo |
| 22 | **SFM mini-bölüm** | Trade-off matrisi + Class 4 Test2'de SFM lider | manuel tablo + bar chart |
| 23 | **Ensemble şeması** | 3-seed softmax averaging görsel | **manuel diyagram (Task 11)** |
| 24 | SOTA savunma | Metodolojik fark + PA/MCA karşılaştırma | `literature_table.md` |
| 25 | Limitations 1 | Class 4 Test2 + Original-eval | `LIMITATIONS.md` |
| 26 | Limitations 2 | Tek-volüm + K-fold + tam ablation | `LIMITATIONS.md` |
| 27 | Future Work | Heterojen ensemble + DA + original-eval + cross-volume | manuel |
| 28 | Teşekkür + soru | İletişim bilgisi | — |

**Tahmini toplam:** 28 slayt × ~1.4 dk = 39 dk (1 dk soru için kalır, ek Q&A süresi varsayılıyor).

---

## 🎯 Prova Notları

- **Prova zamanı:** En az 3 tam prova yap, zaman tut
- **Hız:** Slayt başına ~1.4 dk — hızlı geçme, atlamaya gerek yok
- **Vurgu noktaları:** (a) Methodology fix dürüstlüğü (slayt 15), (b) Single-seed bias bulgusu (slayt 20-21), (c) SFM trade-off (slayt 22), (d) SOTA savunma (slayt 24), (e) Limitations (slayt 25-26)
- **Beklenen sorular:** En az S1, S6, S7, S10, S11 sorulacak — cevapları ezberle
- **Bekleme cümlesi:** Soru gelince "İyi soru" deme, doğrudan cevaba gir
- **Kanonik sayılar dosyası:** [`SAYILAR_KANONIK.md`](SAYILAR_KANONIK.md) — provadan önce ezberle, yanlış sayı söyleme

**Önemli:** Her bölüm sonunda **slayt geçişini sözel köprü** ile yap. "Bu konuya geldikten sonra şuna bakalım" tarzı geçişler dinleyiciyi tutarsız hissettirmez.

**Anahtar mesajlar (sunum sonunda dinleyicinin aklında kalması gereken):**
1. "Combined mIoU 0.791 ± 0.006" — ana sayı, varyansla
2. "Methodology hatalarını gizlemek yerine dürüstçe düzelttim" — akademik duruş
3. "Single-seed sonuç yanıltıcı olabilir — multi-seed gerekli" — pedagojik bulgu
4. "Class 4 Test2 hâlâ açık problem — alan adaptasyonu gerekli" — gelecek yön
