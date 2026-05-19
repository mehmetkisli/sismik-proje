# PC Görevi — Original-Resolution Evaluator Koşumu

> **Bu dosya Mac'te hazırlandı.** PC'de Claude Code aç, projeyi pull et, ve aşağıdaki adımları sırayla uygula. Sorun çıkarsa devam etme, durup raporla.

---

## Neden

`scripts/eval_originalres.py` v9 ensemble için **orijinal çözünürlükte** (701×255 / 401×255) metrik hesaplayan değerlendiriciyi içerir. Bu, **Alaudah 2019 baseline** ile birebir kıyas yapabilmek için gerekli (Alaudah orijinal çözünürlükte değerlendirir; bizim canonical metrikler 384×384 resized space'te). Sayısal olarak Alaudah'ı PA/MCA/FwIoU üzerinde geçtik ama formal "geçtim" iddiası bu evaluator olmadan yapılmıyor — şimdi bu boşluğu kapatıyoruz.

---

## Adımlar (sırayla, atlamadan)

### 1. Son hâli çek

```bash
git pull
```

Beklenen: `be6ed60` veya sonrası commit görmeli. Bu dosyanın PC'de görünmesi pull başarılı demektir.

### 2. Checkpoint'leri doğrula

```bash
ls checkpoints_v7/*.pth        # Linux/Mac stili
# veya Windows PowerShell:
dir checkpoints_v7\*.pth
```

Şu 3 dosya görünmeli:
- `deeplabv3plus_v9_best.pth` (SEED 42)
- `v9_seed_43_best.pth` (SEED 43)
- `v9_seed_44_best.pth` (SEED 44)

**Eksik varsa:** dur, kullanıcıya dosyaların gerçek konumunu sor.

### 3. Conda env'i doğrula

```bash
conda env list
```

`sismik` adında env olmalı. Yoksa env adını öğren ve aşağıdaki komutu güncelle.

### 4. Evaluator'ı koştur

```bash
conda run -n sismik python scripts/eval_originalres.py 2>&1 | tee eval_originalres.log
```

**Beklenen süre:** ~15-30 dakika (3 model × 2 test seti × multi-scale TTA).

**Olası sorunlar:**
- **OOM (RAM):** `--seeds 42` ile başla, sonra `--seeds 43`, sonra `--seeds 44` — her birinde önceki softmax'ları sil
- **VRAM yetersiz:** `--batch-size 2` veya `--batch-size 1` ekle
- **Conda env adı farklı:** komutu uyarla, kullanıcıya hangi env'i kullandığını bildir

### 5. Çıktıyı doğrula

```bash
ls results/metrics/v9_ensemble_originalres_metrics.json
```

Olmalı.

### 6. Sayıları konsola dök

```bash
python -c "
import json
d = json.load(open('results/metrics/v9_ensemble_originalres_metrics.json'))
e = d['ensemble']
print('=== ENSEMBLE — ORIGINAL RESOLUTION ===')
print(f\"Combined  mIoU={e['combined']['mIoU']*100:.2f}%  FwIoU={e['combined']['FwIoU']*100:.2f}%  PA={e['combined']['PA']*100:.2f}%  MCA={e['combined']['MCA']*100:.2f}%\")
print(f\"Test1     mIoU={e['test1']['mIoU']*100:.2f}%  FwIoU={e['test1']['FwIoU']*100:.2f}%\")
print(f\"Test2     mIoU={e['test2']['mIoU']*100:.2f}%  FwIoU={e['test2']['FwIoU']*100:.2f}%\")
print(f\"Class 4 Test2 IoU: {e['test2']['per_class_iou'][4]*100:.2f}%\")
print()
print('=== ALAUDAH 2019 BASELINE (referans) ===')
print('PA=90.5%  MCA=81.7%  FwIoU=83.2%')
print()
print('=== KARSILASTIRMA ===')
print(f\"PA delta:    {(e['combined']['PA']-0.905)*100:+.2f}p\")
print(f\"MCA delta:   {(e['combined']['MCA']-0.817)*100:+.2f}p\")
print(f\"FwIoU delta: {(e['combined']['FwIoU']-0.832)*100:+.2f}p\")
"
```

### 7. Commit + push

```bash
git add results/metrics/v9_ensemble_originalres_metrics.json eval_originalres.log
git commit -m "Original-res evaluator sonuclari: Alaudah birebir kiyas"
git push
```

### 8. Son rapor (kullanıcıya)

Şu üç soruyu cevapla:

1. **Resized vs original-res:** Hangi metrik ne kadar değişti? (örnek: `Combined mIoU 0.791 → 0.XXX, Δ +/-X.Xp`)
2. **Alaudah karşısında:** Üç metrikte de hâlâ geçiyor muyuz? Pozitif delta'lar var mı?
3. **Formal iddia mümkün mü?** Eğer üç metrikte de geçiyorsak ve mIoU değişimi minimal ise (örn. <2 puan), sunumda "Alaudah'ı geçtim, aynı evaluator protokolünde" diyebiliriz.

---

## Bittiğinde

Kullanıcı Mac'e döndüğünde `git pull` çekip yeni sayıları aldıktan sonra:
- `SAYILAR_KANONIK.md`'ye orijinal-res sayılar eklenecek
- `SUNUM_SCRIPTI.md` savunma cümlesi "formal iddia mümkün" hâline güncellenecek
- `sunum_calisma_kilavuzu.html` Slayt #07 güncellenecek

Bu işi Mac'teki Claude Code yapacak — sen sadece sayıları push et yeter.
