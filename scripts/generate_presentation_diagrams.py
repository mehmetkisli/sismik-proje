"""
Sunum için 4 manuel diyagram üretir (matplotlib, GPU gerektirmez).

Çıktılar:
- results/figures/2_5d_input_schema.png         (2.5D girdi şeması)
- results/figures/methodology_fix_schema.png    (Önce/sonra split şeması)
- results/figures/deeplab_block_diagram.png     (Mimari blok diyagramı)
- results/figures/ensemble_schema.png           (3-seed softmax averaging)

Çalıştırma (kök dizinden):
    venv/bin/python scripts/generate_presentation_diagrams.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1a3a6c"
ACCENT = "#d97706"
GREEN = "#2f9e44"
RED = "#c92a2a"
YELLOW = "#f59f00"
GRAY = "#868e96"
LIGHT = "#f1f3f5"


# ---------------------------------------------------------------------------
# 1) 2.5D Girdi Şeması
# ---------------------------------------------------------------------------
def make_2_5d_schema():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    rng = np.random.default_rng(7)
    slice_labels = ["i-2", "i-1", "i", "i+1", "i+2"]
    base_x = 0.6
    for k, lbl in enumerate(slice_labels):
        x0 = base_x + k * 0.95
        is_center = k == 2
        # her slice'a düşük çözünürlüklü sismik benzeri bir doku ekle
        h, w = 24, 16
        bg = np.cumsum(rng.normal(0, 1, size=(h, w)), axis=0)
        bg = (bg - bg.min()) / (bg.max() - bg.min())
        ax.imshow(
            bg,
            extent=(x0, x0 + 0.75, 0.9, 4.1),
            cmap="RdBu_r",
            aspect="auto",
            alpha=0.85,
        )
        rect = Rectangle(
            (x0, 0.9),
            0.75,
            3.2,
            linewidth=2.5 if is_center else 1.2,
            edgecolor=ACCENT if is_center else NAVY,
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + 0.375,
            4.35,
            lbl,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold" if is_center else "normal",
            color=ACCENT if is_center else NAVY,
        )

    ax.text(
        base_x + 5 * 0.95 / 2 - 0.1,
        0.3,
        "5 komşu inline kesiti",
        ha="center",
        fontsize=12,
        color=GRAY,
    )

    arrow = FancyArrowPatch(
        (5.7, 2.5), (7.5, 2.5), arrowstyle="->,head_width=8,head_length=12",
        lw=2.5, color=NAVY,
    )
    ax.add_patch(arrow)
    ax.text(6.6, 2.85, "stack", ha="center", fontsize=11, color=NAVY, style="italic")

    # 5-kanallı stack görselleştirmesi
    channel_colors = ["#74c0fc", "#4dabf7", "#339af0", "#1c7ed6", "#1864ab"]
    for k in range(5):
        offset = k * 0.16
        rect = Rectangle(
            (7.9 + offset, 1.1 + offset),
            2.0,
            2.6,
            linewidth=1.5,
            edgecolor=NAVY,
            facecolor=channel_colors[k],
            alpha=0.85,
            zorder=10 - k,
        )
        ax.add_patch(rect)
    ax.text(
        7.9 + 2.5 * 0.16 + 1.0,
        4.4,
        "5 kanal",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        7.9 + 2.5 * 0.16 + 1.0,
        0.6,
        "(H × W × 5)",
        ha="center",
        fontsize=11,
        color=GRAY,
        style="italic",
    )

    arrow2 = FancyArrowPatch(
        (10.6, 2.5), (11.5, 2.5), arrowstyle="->,head_width=8,head_length=12",
        lw=2.5, color=NAVY,
    )
    ax.add_patch(arrow2)
    ax.text(
        11.7,
        2.5,
        "DeepLabV3+\nencoder\n(5→3 ch adapt)",
        ha="left",
        va="center",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )

    ax.set_title(
        "2.5D Girdi: 5 Komşu Slice → 5 Kanallı Görüntü",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    fig.tight_layout()
    fig.savefig(OUT / "2_5d_input_schema.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 2_5d_input_schema.png")


# ---------------------------------------------------------------------------
# 2) Methodology Fix Şeması
# ---------------------------------------------------------------------------
def make_methodology_fix_schema():
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # üst panel: v7-broken
    ax = axes[0]
    ax.set_xlim(0, 401)
    ax.set_ylim(-0.5, 2)
    ax.set_yticks([])
    ax.set_xlabel("Inline indeksi", fontsize=11)
    ax.set_title(
        "v7-broken: Üç sızıntı",
        fontsize=14,
        fontweight="bold",
        color=RED,
        loc="left",
        pad=8,
    )

    ax.add_patch(Rectangle((0, 0), 320, 1, facecolor=GREEN, alpha=0.6, edgecolor="black"))
    ax.text(160, 0.5, "TRAIN (inline 0–319, %80)", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.add_patch(Rectangle((320, 0), 81, 1, facecolor=RED, alpha=0.7, edgecolor="black"))
    ax.text(360, 0.5, "VAL\n(320–400)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")

    # 3 sızıntı annotation
    ax.annotate(
        "❶ Contiguous val bias\n(lokasyon yanlılığı)",
        xy=(360, 1.0), xytext=(310, 1.7),
        fontsize=10, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )
    ax.annotate(
        "❷ Xline leakage\n(601 xline'ın hepsi train,\nval pikselleri içerir)",
        xy=(200, 0), xytext=(70, -0.4),
        fontsize=9, color=RED, fontweight="bold", ha="left",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )
    ax.annotate(
        "❸ 2.5D komşu sızıntı\n(val sınırı kanalları train'den)",
        xy=(320, 0.5), xytext=(220, 1.7),
        fontsize=10, color=RED, fontweight="bold", ha="left",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )

    # alt panel: v7-fixed (Yol A)
    ax = axes[1]
    ax.set_xlim(0, 401)
    ax.set_ylim(-0.5, 2)
    ax.set_yticks([])
    ax.set_xlabel("Inline indeksi", fontsize=11)
    ax.set_title(
        "v7-fixed (Yol A): Val ortada + buffer + xline cropped",
        fontsize=14,
        fontweight="bold",
        color=GREEN,
        loc="left",
        pad=8,
    )

    ax.add_patch(Rectangle((0, 0), 158, 1, facecolor=GREEN, alpha=0.6, edgecolor="black"))
    ax.text(79, 0.5, "TRAIN-A", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.add_patch(Rectangle((158, 0), 2, 1, facecolor=YELLOW, alpha=0.9, edgecolor="black"))
    ax.add_patch(Rectangle((160, 0), 80, 1, facecolor=RED, alpha=0.7, edgecolor="black"))
    ax.text(200, 0.5, "VAL\n(160–239)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.add_patch(Rectangle((240, 0), 2, 1, facecolor=YELLOW, alpha=0.9, edgecolor="black"))
    ax.add_patch(Rectangle((242, 0), 159, 1, facecolor=GREEN, alpha=0.6, edgecolor="black"))
    ax.text(321, 0.5, "TRAIN-B", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")

    ax.annotate(
        "Buffer ±2\n(2.5D komşu sızıntı önlenir)",
        xy=(159, 1.0), xytext=(50, 1.7),
        fontsize=10, color=NAVY, fontweight="bold", ha="left",
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2),
    )
    ax.annotate(
        "Buffer ±2",
        xy=(241, 1.0), xytext=(260, 1.65),
        fontsize=10, color=NAVY, fontweight="bold", ha="left",
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2),
    )
    ax.text(
        200, -0.35,
        "Xline image'ları train_inline_mask ile val pikselleri CROPPED → 3D leakage kapalı",
        ha="center", fontsize=10, color=NAVY, style="italic", fontweight="bold",
    )

    legend_elements = [
        mpatches.Patch(facecolor=GREEN, alpha=0.6, label="Train"),
        mpatches.Patch(facecolor=RED, alpha=0.7, label="Validation"),
        mpatches.Patch(facecolor=YELLOW, alpha=0.9, label="Buffer (±2 inline)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10, frameon=False,
               bbox_to_anchor=(0.98, 1.0))

    fig.suptitle(
        "Methodology Fix: 3 Veri Sızıntısı → Yol A Düzeltmesi",
        fontsize=16, fontweight="bold", color=NAVY, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "methodology_fix_schema.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] methodology_fix_schema.png")


# ---------------------------------------------------------------------------
# 3) DeepLabV3+ Blok Diyagramı
# ---------------------------------------------------------------------------
def make_deeplab_diagram():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, color=NAVY, fontsize=11, text_color="white"):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor=NAVY, facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color,
        )

    def arrow(x1, y1, x2, y2, color=NAVY, lw=2.0):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="->,head_width=6,head_length=10",
                lw=lw, color=color,
            )
        )

    # Girdi
    box(0.3, 2.5, 1.6, 1.4, "Girdi\n5×H×W\n(2.5D)", color="#dde3ea", text_color=NAVY)
    arrow(1.95, 3.2, 2.55, 3.2)

    # EfficientNet-B4 encoder (stages)
    box(2.6, 4.1, 1.6, 0.9, "Stem (5→3 adapt)", color=NAVY, fontsize=10)
    box(2.6, 3.0, 1.6, 0.9, "EfficientNet-B4\nStages 1-4", color=NAVY, fontsize=10)
    box(2.6, 1.9, 1.6, 0.9, "Low-level features\n(stage 2 çıkışı)", color="#4dabf7", fontsize=9)
    arrow(3.4, 4.05, 3.4, 3.95)
    arrow(3.4, 2.95, 3.4, 2.85)

    ax.text(3.4, 5.15, "Encoder", ha="center", fontsize=12,
            fontweight="bold", color=NAVY)

    arrow(4.25, 3.45, 4.85, 3.45)

    # ASPP
    aspp_x = 4.9
    box(aspp_x, 4.6, 1.8, 0.6, "1×1 conv", color="#1864ab", fontsize=9)
    box(aspp_x, 3.85, 1.8, 0.6, "Atrous r=12", color="#1864ab", fontsize=9)
    box(aspp_x, 3.10, 1.8, 0.6, "Atrous r=24", color="#1864ab", fontsize=9)
    box(aspp_x, 2.35, 1.8, 0.6, "Atrous r=36", color="#1864ab", fontsize=9)
    box(aspp_x, 1.60, 1.8, 0.6, "Image Pool", color="#1864ab", fontsize=9)

    ax.text(aspp_x + 0.9, 5.4, "ASPP", ha="center", fontsize=12,
            fontweight="bold", color=NAVY)
    ax.text(aspp_x + 0.9, 1.2, "(çoklu ölçek bağlam)",
            ha="center", fontsize=9, color=GRAY, style="italic")

    # Concat
    box(7.0, 3.0, 1.0, 0.9, "Concat\n+ 1×1", color=ACCENT, fontsize=9)
    for y in [4.9, 4.15, 3.40, 2.65, 1.90]:
        arrow(aspp_x + 1.8, y, 7.0, 3.45, lw=1.0)
    arrow(8.05, 3.45, 8.6, 3.45)

    # Decoder
    box(8.65, 3.0, 2.0, 0.9, "Decoder\n(upsample ×4)", color=NAVY, fontsize=10)
    # Low-level skip
    ax.add_patch(
        FancyArrowPatch(
            (4.2, 2.35), (8.65, 2.7),
            connectionstyle="arc3,rad=-0.25",
            arrowstyle="->,head_width=6,head_length=10",
            lw=1.6, color="#4dabf7",
        )
    )
    ax.text(6.4, 1.5, "Skip connection\n(low-level features)",
            fontsize=9, color="#1864ab", style="italic", ha="center")

    arrow(10.7, 3.45, 11.25, 3.45)
    # Üst (×4) upsample
    box(11.3, 3.0, 1.5, 0.9, "Upsample\n×4", color=NAVY, fontsize=10)
    arrow(12.85, 3.45, 13.4, 3.45)

    # Çıktı
    box(13.45, 2.5, 0.55, 1.4, "Out\n6×H×W", color="#dde3ea", text_color=NAVY, fontsize=9)

    ax.set_title(
        "DeepLabV3+ Mimari Bloğu (EfficientNet-B4 backbone, ASPP rates 12/24/36)",
        fontsize=14, fontweight="bold", color=NAVY, pad=15,
    )

    fig.tight_layout()
    fig.savefig(OUT / "deeplab_block_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] deeplab_block_diagram.png")


# ---------------------------------------------------------------------------
# 4) 3-Seed Ensemble + Softmax Averaging Şeması
# ---------------------------------------------------------------------------
def make_ensemble_schema():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, label, color=NAVY, fontsize=11, text_color="white"):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor=NAVY, facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color,
        )

    def arrow(x1, y1, x2, y2, color=NAVY, lw=1.8):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="->,head_width=6,head_length=10",
                lw=lw, color=color,
            )
        )

    # Girdi
    box(0.3, 2.9, 1.6, 1.0, "Test girdisi\n(inline / xline)",
        color="#dde3ea", text_color=NAVY, fontsize=10)

    # 3 model (seed)
    seed_data = [
        ("SEED 42", "v9", 5.0, "Combined mIoU\n0.7769"),
        ("SEED 43", "v9", 3.2, "Combined mIoU\n0.7850"),
        ("SEED 44", "v9", 1.4, "Combined mIoU\n0.7920"),
    ]
    for label, sub, y, metric in seed_data:
        box(2.3, y, 2.0, 1.0, f"{label}\n({sub} mimari)",
            color=NAVY, fontsize=11)
        arrow(1.95, 3.4, 2.3, y + 0.5)
        # softmax çıktısı
        box(4.6, y, 1.6, 1.0, "softmax\n(6×H×W)",
            color="#4dabf7", text_color="white", fontsize=10)
        arrow(4.3, y + 0.5, 4.6, y + 0.5)
        ax.text(7.4, y + 0.5, metric,
                fontsize=9, color=GRAY, ha="center", va="center", style="italic")
        arrow(6.25, y + 0.5, 8.45, 3.4, lw=1.4)

    # Softmax averaging
    box(8.5, 2.9, 1.9, 1.0, "Softmax\nORTALAMA",
        color=ACCENT, text_color="white", fontsize=11)
    arrow(10.45, 3.4, 10.95, 3.4)

    # Argmax
    box(11.0, 2.9, 1.4, 1.0, "argmax", color=NAVY, fontsize=11)
    arrow(12.45, 3.4, 12.85, 3.4)

    # Final
    box(12.9, 2.9, 0.05, 1.0, "", color="#dde3ea")
    ax.text(12.4, 4.7,
            "Final tahmin\n(6-sınıf maske)",
            fontsize=10, color=NAVY, fontweight="bold", ha="left")

    # Sonuç vurgusu
    ax.text(6.5, 0.65,
            "ENSEMBLE Combined mIoU = 0.7910  ($+0.6$ puan vs mean tek-seed)\n"
            "Std (3 seed) = ±0.006   →   varyans ölçülmüş, dürüst raporlama",
            ha="center", fontsize=12, color=ACCENT, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff3cd",
                      ec=ACCENT, lw=1.5))

    ax.set_title(
        "3-Seed Multi-Seed Ensemble (Softmax Averaging)",
        fontsize=15, fontweight="bold", color=NAVY, pad=15,
    )

    fig.tight_layout()
    fig.savefig(OUT / "ensemble_schema.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[OK] ensemble_schema.png")


if __name__ == "__main__":
    print("Diyagramlar üretiliyor →", OUT)
    make_2_5d_schema()
    make_methodology_fix_schema()
    make_deeplab_diagram()
    make_ensemble_schema()
    print("Tamamlandı.")
