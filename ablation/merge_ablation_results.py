"""Tüm ablation sonuçlarını oku, karşılaştırma tablosu üret.

GPU işleri bittikten sonra çalıştırılır. Çıktıyı script + slayda kopyalanmak üzere
hem markdown hem JSON formatında verir.

Kullanım:
  venv/bin/python ablation/merge_ablation_results.py

Çıktı:
  results/metrics/ablation_summary.json     (programatik kullanım için)
  results/metrics/ablation_summary.md       (script + slayda kopyalanabilir tablo)
"""
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
METRICS_DIR = PROJECT_DIR / "results" / "metrics"

# Karşılaştırma için baseline: v9 single-seed (SEED=42, mevcut)
BASELINE_JSON = METRICS_DIR / "deeplabv3plus_v9_metrics.json"


def load(p):
    if not p.exists():
        print(f"[WARN] {p.name} yok — atlanıyor")
        return None
    with open(p) as f:
        return json.load(f)


def fmt_pp(x):
    return f"{x*100:+.2f}p"


def fmt_pct(x):
    return f"{x*100:.2f}%"


def main():
    baseline = load(BASELINE_JSON)
    if baseline is None:
        print(f"HATA: Baseline v9 metric JSON'u bulunamadı: {BASELINE_JSON}")
        return

    # v9 baseline değerleri (TTA on)
    b_comb = baseline["combined"]["mIoU"]
    b_t1 = baseline["test1"]["mIoU"]
    b_t2 = baseline["test2"]["mIoU"]
    b_c4 = baseline["test2"]["per_class_iou"][4]
    b_val = baseline.get("best_val_miou") or max(baseline.get("history", {}).get("val_miou", [0]))

    print(f"Baseline (v9 single-seed SEED=42):")
    print(f"  Combined {fmt_pct(b_comb)} | Test1 {fmt_pct(b_t1)} | Test2 {fmt_pct(b_t2)} | "
          f"C4 Test2 {fmt_pct(b_c4)} | Best val {fmt_pct(b_val)}")

    # 1) Training ablation'ları
    ablation_tags = ["nolovasz", "ch3", "noxlineaware"]
    label_map = {
        "nolovasz": "Lovász OFF (TripleLoss)",
        "ch3": "3-channel 2.5D (±1)",
        "noxlineaware": "xline-aware aug OFF",
    }

    train_rows = []
    train_rows.append({
        "config": "v9 baseline (SEED=42)",
        "loss": "Quadruple",
        "channels": 5,
        "xline_aware": True,
        "combined": b_comb,
        "test1": b_t1,
        "test2": b_t2,
        "c4_test2": b_c4,
        "best_val": b_val,
        "delta_combined": 0.0,
    })

    for tag in ablation_tags:
        p = METRICS_DIR / f"ablation_{tag}_metrics.json"
        data = load(p)
        if data is None:
            train_rows.append({
                "config": f"{label_map[tag]} (TBD)",
                "loss": "?", "channels": "?", "xline_aware": "?",
                "combined": None, "test1": None, "test2": None,
                "c4_test2": None, "best_val": None, "delta_combined": None,
            })
            continue
        flags = data["ablation_flags"]
        train_rows.append({
            "config": label_map[tag],
            "loss": "Triple" if flags["no_lovasz"] else "Quadruple",
            "channels": flags["channels"],
            "xline_aware": not flags["no_xline_aware"],
            "combined": data["combined"]["mIoU"],
            "test1": data["test1"]["mIoU"],
            "test2": data["test2"]["mIoU"],
            "c4_test2": data["test2"]["per_class_iou"][4],
            "best_val": data.get("best_val_miou"),
            "delta_combined": data["combined"]["mIoU"] - b_comb,
        })

    # 2) TTA ablation
    tta_data = load(METRICS_DIR / "ablation_tta_variants.json")
    tta_rows = []
    if tta_data is not None:
        for mode in ("none", "hflip", "multiscale"):
            r = tta_data["results"][mode]
            tta_rows.append({
                "config": mode,
                "description": r["description"],
                "combined": r["combined"]["mIoU"],
                "test1": r["test1"]["mIoU"],
                "test2": r["test2"]["mIoU"],
                "c4_test2": r["test2"]["per_class_iou"][4],
                "delta_vs_multiscale": tta_data["summary"][mode]["delta_vs_multiscale"],
            })

    # Çıktılar
    out_json = METRICS_DIR / "ablation_summary.json"
    with open(out_json, "w") as f:
        json.dump({"training": train_rows, "tta": tta_rows,
                   "baseline_combined_miou": b_comb}, f, indent=2)
    print(f"\n→ {out_json}")

    # Markdown
    md_lines = []
    md_lines.append("# Ablation Sonuçları — v9 Mimarisi\n")
    md_lines.append("Baseline: **v9 single-seed (SEED=42)** — `deeplabv3plus_v9_metrics.json`\n")
    md_lines.append("Multi-scale TTA + xline-aware aug + Lovász + 5-channel **ON** (tam pipeline).\n")

    md_lines.append("\n## 1. Training Ablation (mimari/loss bileşenleri)\n")
    md_lines.append("| Konfig | Loss | Ch | Xline-aware | Best val | Test1 | Test2 | **Combined** | Δ vs v9 | C4 Test2 |")
    md_lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in train_rows:
        comb_str = fmt_pct(r["combined"]) if r["combined"] is not None else "TBD"
        delta_str = fmt_pp(r["delta_combined"]) if r["delta_combined"] is not None else "—"
        val_str = fmt_pct(r["best_val"]) if r["best_val"] is not None else "TBD"
        t1_str = fmt_pct(r["test1"]) if r["test1"] is not None else "TBD"
        t2_str = fmt_pct(r["test2"]) if r["test2"] is not None else "TBD"
        c4_str = fmt_pct(r["c4_test2"]) if r["c4_test2"] is not None else "TBD"
        xline_str = "ON" if r["xline_aware"] is True else ("OFF" if r["xline_aware"] is False else "?")
        md_lines.append(f"| {r['config']} | {r['loss']} | {r['channels']} | {xline_str} | "
                       f"{val_str} | {t1_str} | {t2_str} | **{comb_str}** | {delta_str} | {c4_str} |")

    md_lines.append("\n## 2. TTA Varyant Ablation (inference-only)\n")
    if tta_rows:
        md_lines.append("| Mod | Açıklama | Test1 | Test2 | **Combined** | Δ vs multiscale | C4 Test2 |")
        md_lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in tta_rows:
            flag = " ← v9 default" if r["config"] == "multiscale" else ""
            md_lines.append(f"| {r['config']} | {r['description']}{flag} | "
                           f"{fmt_pct(r['test1'])} | {fmt_pct(r['test2'])} | "
                           f"**{fmt_pct(r['combined'])}** | {fmt_pp(r['delta_vs_multiscale'])} | "
                           f"{fmt_pct(r['c4_test2'])} |")
    else:
        md_lines.append("⚠️ TTA ablation henüz çalıştırılmadı (`ablation/eval_tta_variants.py`).\n")

    md_lines.append("\n## 3. Yorumlama Notları\n")
    md_lines.append("- **Lovász katkısı:** TripleLoss baseline'a vs Quadruple delta = Lovász'ın net katkısı.")
    md_lines.append("- **5-ch katkısı:** 3-channel vs 5-channel — özellikle C4 Test2'de fark öne çıkar (anisotropic morfoloji).")
    md_lines.append("- **xline-aware aug katkısı:** OFF konfigi xline genelleme zayıflığını ölçer — Test2 mIoU farkı kritik.")
    md_lines.append("- **TTA katkısı:** Multi-scale TTA, HFlip-only ve no-TTA arasındaki delta → inference-time augmentation faydası.")
    md_lines.append("- Class 4 Test2 hassasiyeti her ablation'da ayrı raporlanmalı — single-seed varyans yüksek (std ~0.04).")

    out_md = METRICS_DIR / "ablation_summary.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"→ {out_md}")

    print("\nKopyalanabilir tablo:\n")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
