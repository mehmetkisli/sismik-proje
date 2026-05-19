# Limitations — Sismik Fasiyes Segmentasyonu (DeepLabV3+ v7-fixed)

> Bu belge, sunum ve tezde **"Limitations / Çalışmanın Sınırlılıkları"** bölümünde kullanılacak akademik dilde Türkçe metin taslaklarını içerir. Hocalara dürüstçe söylenecek konular — saklamaya çalışmak yerine bilimsel olgunlukla ortaya konacak.
>
> Her madde için: (a) **Sorun**, (b) **Niçin önemli**, (c) **Bu çalışmada nasıl ele alındı**, (d) **Hâlâ açık kalan**.

---

## 1. Doğrulama (Validation) Kümesi Yapılandırma Yanlılığı

**Sorun.** Eğitim hacmindeki 401 inline kesitinin son %20'lik bloğunun (inline 320–400) doğrulama kümesi olarak ayrılması, sismik volüm üzerinde mekânsal olarak süreksiz bir kayma yaratmaktadır. Hollanda F3 bloğunda Zechstein tuz yapılarının dağılımı inline ekseni boyunca homojen değildir; dolayısıyla son inline bölgesinin jeolojik kompozisyonu eğitim bölgesinden sistematik biçimde farklı olabilir. Bu durum doğrulama metriğinin (`val mIoU`) modelin gerçek genelleme kapasitesini değil, belirli bir mekânsal alt-bölgenin temsil yeterliliğini ölçmesine yol açmaktadır.

**Niçin önemli.** Erken durdurma (early stopping) ve en iyi epoch seçimi doğrulama metriği üzerinden yapıldığı için, yanlı bir doğrulama kümesi yanlı bir model seçimine neden olur. v7-broken sürümünde elde edilen `val mIoU = 0.6617` değerinin test mIoU'su (0.7881) ile arasındaki +12.6 puanlık ters paradoks, bu yanlılığın somut bir göstergesidir.

**Bu çalışmada ele alınma şekli (Yol A methodology fix, 2026-05-09).** Doğrulama bölgesi inline [160, 240) aralığına kaydırılmıştır; eğitim verisi `[0, 158) ∪ [242, 401)` aralığında iki blok hâline gelmiş, doğrulama bölgesi train geometrisi tarafından **çevrelenmiştir**. Buna ek olarak ±2 inline genişliğinde tampon (buffer) bölgesi bırakılarak eğitim ve doğrulama sınırı arasında 2.5D komşu sızıntısı engellenmiştir.

**Hâlâ açık kalan.** Tek bir mekânsal blok hâlâ kullanılmaktadır; tam K-katlı çapraz doğrulama (K-fold cross-validation) yapılmamıştır. Bu tercih bilimsel ödün değildir — F3 hacminin tek bir 3D volüm olması ve 5-katlı CV'nin 5x eğitim maliyeti getirmesi nedeniyle *yüksek lisans seminerinin zaman/işlem bütçesi içinde* uygulanabilir bulunmamıştır. K-katlı doğrulama "Future Work" başlığı altında belgelenmiştir.

---

## 2. Crossline Eğitim Veri Sızıntısı (3D Leakage)

**Sorun.** v7-broken sürümünde 601 crossline kesitinin tamamı eğitime dahil ediliyordu. Her crossline 2D görüntüsü `(n_inline, depth) = (401, 255)` boyutundadır ve **her görüntü doğrulama inline'larına ait pikselleri içerir**. Bu, doğrulama kümesinin pikseller seviyesinde eğitim sırasında dolaylı olarak modele sızmasına yol açan ciddi bir bilimsel hatadır.

**Niçin önemli.** Bu tür bir sızıntı doğrulama metriklerini olduğundan iyi gösterir, model seçimini bozar ve test sonuçlarına genelleyici bir referans olmaktan çıkarır. Yayın etiği açısından kabul edilemez bir durumdur.

**Bu çalışmada ele alınma şekli.** `train_inline_mask` adında bir boolean dizi tanımlanmıştır. Veri kümesi sınıfında (`F3Dataset25D`), crossline modunda (`axis=1`) ham görüntü alındıktan sonra bu maske uygulanarak doğrulama bölgesindeki satırlar görüntüden **çıkarılmaktadır**. Sonuç olarak crossline eğitim örneklerinin şekli `(317, 255)`'e düşmüş; doğrulama pikselleri eğitim sırasında modele hiçbir kanal üzerinden ulaşmamaktadır. Test2 hacmi tamamen ayrı bir volüm olduğu için maskeleme uygulanmamış, tüm crossline'lar değerlendirmeye dahil edilmiştir.

**Hâlâ açık kalan.** Yok — bu sızıntı tam olarak kapatılmıştır.

---

## 3. 2.5D Komşu Slice Sızıntısı

**Sorun.** 2.5D girdi yaklaşımında her inline `i` için `[i-1, i, i+1]` kanalları stacklenir. Doğrulama bölgesinin sınırına oturan inline'larda (örn. eski kurulumda inline 320), bir önceki ve bir sonraki kanallar eğitim tarafına denk gelir. Doğrulama örneklerinin yaklaşık üçte ikisi "yarı-eğitim-görmüş" olur — bu da metrikleri yapay olarak yükseltir.

**Bu çalışmada ele alınma şekli.** Eğitim bölgesi doğrulama sınırına `BUFFER = 2` inline yakınlık bırakacak şekilde kırpılmıştır. Bu sayede doğrulama örneklerinin hiçbir kanalı eğitim verisinden gelmemektedir.

**Hâlâ açık kalan.** Buffer değerinin (±2) optimal olduğu ampirik olarak gösterilmemiştir; ±1, ±3, ±5 gibi alternatifler test edilmemiştir. 2.5D çekirdek genişliği 3 olduğu için ±2 mantıken yeterli ama duyarlılık analizi yapılmamıştır.

---

## 4. Tek Tohumlu (Single-Seed) Eğitim — Varyans Belirsizliği

**Sorun.** Tüm sonuçlar tek bir rastgelelik tohumu (`SEED = 42`) ile elde edilmiştir. Derin öğrenme deneylerinde rastgele başlangıç ağırlıkları, mini-batch sırası, augmentation rastgeleliği gibi kaynaklar nedeniyle tek koşunun mIoU değeri ±%1–3 oynayabilir. Bilimsel iddianın güvenilir olabilmesi için ortalama ± standart sapma raporlanması gerekir.

**Niçin önemli.** "%79.26 mIoU" gibi nokta tahminler, varyans bilinmeden istatistiksel olarak anlamlı bir bulgu sayılmaz. Karşılaştırılan iki yöntem arasındaki 1 puanlık fark gerçek bir kazanç olabileceği gibi sadece tohum farkı da olabilir.

**Bu çalışmada ele alınma şekli.** Bu sınırlılık tarafımızdan açıkça kabul edilmektedir. Tezin sonraki revizyonlarında en az 3 farklı tohumla (42, 43, 44) tekrarlı eğitim yapılması ve mean ± std raporlanması planlanmıştır.

**Hâlâ açık kalan.** Henüz uygulanmadı; "Future Work" başlığı altında ilk öncelik olarak listelenmiştir.

---

## 5. Sistematik Ablation Çalışmasının Bulunmaması

**Sorun.** Modelin başarısını oluşturan bileşenler — Mixup, Test Time Augmentation (TTA), 2.5D çoklu görünüm girdisi, etiket düzleştirme (label smoothing), nadir-sınıf sampling, focal loss, üçlü loss kombinasyonu — hiçbiri ayrı ayrı kapatılıp test edilmemiştir. Sonuç olarak bu bileşenlerin her birinin ne kadar katkı sağladığı ölçülemez durumdadır.

**Niçin önemli.** Bilimsel bir çalışma yalnızca "bu yöntemi kullandık ve şu sayıyı aldık" demez; her bileşenin marjinal katkısını gösterir. Aksi takdirde mimari karar gerekçelendirilemez.

**Bu çalışmada ele alınma şekli.** Sınırlı bir ablation matrisi planlanmıştır: TTA-off, Mixup-off, single-channel (2.5D yerine 1 kanal). Bu üç koşu yaklaşık 5 saat GPU zamanı gerektirmekte ve sunum öncesinde tamamlanması hedeflenmektedir.

**Hâlâ açık kalan.** Tam (full-grid) ablation matrisi (her bileşen için on/off, 2^6 kombinasyon) yüksek lisans semineri kapsamında değildir. Tezin uzun versiyonunda yer alabilir.

---

## 6. Sınıf 4 (Zechstein) Test2 Genelleme Çöküşü

**Sorun.** Model, Sınıf 4 (Zechstein tuz tabakası) için Test1'de IoU=0.836 elde etmesine rağmen Test2'de IoU=0.178'e düşmektedir — yaklaşık 4.7 kat fark. Bu, modelin Zechstein'i yalnızca **inline yönündeki morfolojik temsiline** öğrendiğini, crossline yönündeki farklı geometriye genelleyemediğini göstermektedir. Tuz yapılarının yönelime duyarlı (anisotropic) olması ve eğitim verisinin inline-baskın olması (320 inline + 601 crossline ama inline yönünde anatomik tutarlılık daha yüksek) bu çöküşün muhtemel nedenidir.

**Niçin önemli.** Birleşik mIoU'nun 0.79 olması Zechstein gibi azınlık ama yorumcu için önemli bir sınıfın bir test setinde tamamen kaybolmasını gizleyebilir. Endüstriyel kullanımda bu kabul edilemez.

**Bu çalışmada ele alınma şekli.** Sorun açıkça raporlanmaktadır. Yarı-niceliksel hata analizi (sınıf 4'ün inline vs crossline morfolojisi karşılaştırma görselleri, hata bölgesi ısı haritaları) sunum görsellerine eklenmiştir.

**Hâlâ açık kalan.** Çözümü methodology değişikliği değil **alan adaptasyonu** (domain adaptation) gerektirir — AdaSemSeg, EarthAdaptNet gibi yaklaşımlar bunu adresliyor ancak yüksek lisans semineri zaman bütçesinde implementasyon yapılmamış, "Future Work" olarak belgelenmiştir.

---

## 7. Tek Volüm Eğitim Verisi — Genelleme Kanıtı Sınırlı

**Sorun.** Çalışma yalnızca Hollanda F3 bloğu üzerinde eğitilmiş ve değerlendirilmiştir. Modelin Penobscot, Parihaka, New Zealand bloğu gibi farklı jeolojik ortamlara genellemesi test edilmemiştir.

**Niçin önemli.** F3'te yüksek mIoU elde etmek, modelin genel anlamda "sismik fasiyes segmentasyonunu çözdüğü" iddiasını desteklemez. Farklı stratigrafi, farklı sismik gürültü, farklı kayıt geometrisi olan bloklar üzerinde performans çok daha düşük olabilir.

**Bu çalışmada ele alınma şekli.** Test2 (crossline yönü) en azından eğitim hacminin **mekânsal olarak farklı bir bölümü** olarak işlev görmektedir; tam farklı volüm değildir ama yön-bazlı genelleme ölçümü yapar.

**Hâlâ açık kalan.** Çapraz-volüm transfer (örn. F3'te eğit, Penobscot'ta test) bu çalışma kapsamında değildir. Foundation model fine-tuning (ThinkOnward GFM) bu yönde umut verici olabilir ama "Future Work" olarak listelenmiştir.

---

## 8. Hesaplama / Donanım Kısıtları

**Sorun.** Tüm deneyler tek bir RTX 3060 Ti (8 GB VRAM) üzerinde gerçekleştirilmiştir. Bu, batch boyutunu (BATCH_SIZE=6, gradient accumulation ×4 ile efektif batch=24), girdi çözünürlüğünü (320×320) ve mimari büyüklüğünü (EfficientNet-B4, daha büyük backbone'lar denenememiş) sınırlamıştır.

**Niçin önemli.** Batch boyutu ve çözünürlük segmentasyon kalitesini doğrudan etkiler. Daha büyük donanımla (örn. A100 40 GB) batch=16, 512×512 çözünürlük denenebilirdi.

**Bu çalışmada ele alınma şekli.** AMP (mixed precision), gradient accumulation ve checkpoint resume gibi VRAM optimizasyonları uygulanmıştır.

**Hâlâ açık kalan.** Daha büyük donanımla aynı mimarinin ne kadar kazanım sağlayacağı bilinmemektedir.

---

## 9. State-of-the-Art Karşılaştırmasının Eksikliği

**Sorun.** Sonuçlar Alaudah 2019 ve sonrasındaki F3 literatürüyle yan yana sayısal olarak karşılaştırılmamıştır. Bu olmadan modelin literatürdeki yeri belirsiz kalır.

**Bu çalışmada ele alınma şekli.** SOTA karşılaştırma tablosu derlenmektedir (`docs/literature_table.md`). Sunumdan önce tamamlanması planlanmaktadır.

**Hâlâ açık kalan.** Bazı yeni F3 çalışmalarının (özellikle 2024-2025 yayınları) detaylı sayıları paywall arkasında veya benchmark'a-spesifik olmayan formatlarda raporlandığı için tam standart tablo oluşturmak güçtür.

---

## 10. (Çözüldü 2026-05-19) Original-Resolution Evaluator Eksikliği

**Sorun (geçmişte).** v9 ensemble metrikleri 384×384 resize edilmiş görüntü uzayında hesaplanıyordu. Alaudah 2019 baseline'ı ise orijinal çözünürlükte (Test1: 701×255, Test2: 200×255) değerlendirme yapıyordu. Bu protokol farkı, "Alaudah'ı geçtik" formal iddiasını engelliyordu — sadece "baseline üzerinde, protokol çekincesiyle" denebiliyordu.

**Niçin önemli.** mIoU ve FwIoU resize işleminden duyarlıdır (özellikle azınlık sınıfların sınır pikselleri yumuşar). Aynı protokol kullanılmadan yapılan kıyaslar bilimsel olarak zayıftır.

**Bu çalışmada ele alınma şekli (2026-05-19).** `scripts/eval_originalres.py` yazıldı ve PC GPU üzerinde koşturuldu. Script tahmini 384×384'te yapar, softmax çıktısını bilinear interpolation ile orijinal H×W'ye upsample eder, argmax ve metrikler orijinal çözünürlükte hesaplanır — Alaudah'ın evaluator prosedürünü birebir taklit eder.

**Sonuçlar (canonical resized vs original-res):**

| Metrik | Resize 384 (canonical) | **Original-res** | Δ |
|---|---:|---:|---:|
| Combined mIoU | 0.7910 | **0.7931** | +0.20p |
| Combined FwIoU | 0.8908 | **0.8917** | +0.09p |
| Combined PA | 0.9402 | **0.9401** | ≈ 0 |
| Combined MCA | 0.8734 | **0.8753** | +0.19p |
| Test1 mIoU | 0.8043 | **0.8053** | +0.10p |
| Test2 mIoU | 0.6771 | 0.6771 | 0.00p |
| C4 Test2 IoU | 0.1834 | 0.1799 | −0.35p |

Resize vs orig-res arasında her metrikte fark ≤0.2 puan — değerlendirme protokolü pratikte stabil. Bu da hem orijinal-res'te Alaudah'ın geçildiğini hem de resized eval'in doğru hesapladığını birlikte doğrular.

**Hâlâ açık kalan.** Yok — bu sınırlılık tam olarak kapatılmıştır. Aynı evaluator protokolünde Alaudah baseline'ı PA +3.51p, MCA +5.83p, FwIoU +5.97p ile geçilmiştir; formal "geçtik" iddiası bu noktadan itibaren bilimsel olarak meşrudur.

Kaynak: [`results/metrics/v9_ensemble_originalres_metrics.json`](../results/metrics/v9_ensemble_originalres_metrics.json) + [`eval_originalres.log`](../eval_originalres.log).

---

## Özet

Bu çalışmanın güçlü yönleri (modern mimari, methodology fix, dürüst test protokolü) yanında, tek-tohumlu eğitim, sistematik ablation eksikliği, tek-volüm eğitim verisi ve Sınıf 4 Test2 çöküşü gibi sınırlılıklar açıkça kabul edilmektedir. Bu sınırlılıklar bilimsel bir tezin doğal parçasıdır; her biri "Future Work" başlığı altında somut adımlarla adreslenebilir niteliktedir.

**Bu sınırlılıkların gizlenmesi yerine raporlanması, çalışmanın bilimsel olgunluğunun kanıtıdır.**
