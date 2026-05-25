"""Regenerate figures/codec_scores.png from hard-coded table values."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA = {
    "audio-flamingo-2-0.5B": {
        "audiocaps": (0.423, 0.5),
        "audioset": (0.146, 0.159),
        "clotho": (0.553, 0.5494),
        "clotho_aqa": (None, 0.0),
        "mmau": (None, 0.378),
    },
    "audio-flamingo-2-1.5B": {
        "audiocaps": (0.588, 0.645),
        "audioset": (0.195, 0.183),
        "clotho": (0.766, 0.7114),
        "clotho_aqa": (None, 0.001),
        "mmau": (None, 0.437),
    },
    "audio-flamingo-2-3B": {
        "audiocaps": (0.778, 0.753),
        "audioset": (0.436, 0.415),
        "clotho": (0.777, 0.6636),
        "clotho_aqa": (None, 0.384),
        "mmau": (None, 0.685),
    },
    "audio-flamingo-3-hf": {
        "audiocaps": (0.668, 0.578),
        "audioset": (0.122, 0.118),
        "clotho": (0.375, 0.2994),
        "clotho_aqa": (None, 0.0),
        "mmau": (None, 0.192),
    },
}

MODELS = list(DATA.keys())
MODEL_LABELS = ["AF2-0.5B (0.5B)", "AF2-1.5B (1.5B)", "AF2-3B (3.0B)", "AF3-7B (7.0B)"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
MARKERS = ["o", "^", "s", "D"]
DATASETS = ["audiocaps", "audioset", "clotho", "clotho_aqa", "mmau"]
NO_MEMBER = {"clotho_aqa", "mmau"}

# Layout: each dataset block is BLOCK wide; M and NM sit at fixed offsets within it
BLOCK = 2.2
M_OFF = 0.55  # x offset of Member column inside block
NM_OFF = 1.45  # x offset of Non-Member column inside block
# For no-member datasets the NM marker is centred in the block
NM_ONLY_OFF = (M_OFF + NM_OFF) / 2

MARKER_SIZE = 130
JITTER = 0.08  # small y-jitter to separate overlapping markers is not needed here

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_title(
    "CoDeC Normalized Scores  (filled = Member, hollow = Non-Member)",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

bg_colors = ["#d6eaf8", "#fde8d8"]  # light blue / light orange

for i, ds in enumerate(DATASETS):
    x0 = i * BLOCK
    ax.axvspan(x0, x0 + BLOCK, alpha=0.25, color=bg_colors[i % 2], zorder=0)

    has_member = ds not in NO_MEMBER
    nm_x = x0 + (NM_OFF if has_member else NM_ONLY_OFF)

    for j, model in enumerate(MODELS):
        member_score, non_member_score = DATA[model][ds]

        if has_member and member_score is not None:
            ax.scatter(
                x0 + M_OFF,
                member_score,
                marker=MARKERS[j],
                color=COLORS[j],
                s=MARKER_SIZE,
                zorder=3,
            )

        ax.scatter(
            nm_x,
            non_member_score,
            marker=MARKERS[j],
            s=MARKER_SIZE,
            zorder=3,
            facecolors="none",
            edgecolors=COLORS[j],
            linewidths=1.8,
        )

    # Dataset label + M/NM sub-labels
    label_y = -0.06
    ax.text(
        x0 + BLOCK / 2,
        label_y,
        ds,
        ha="center",
        va="top",
        fontsize=10,
        transform=ax.get_xaxis_transform(),
    )
    if has_member:
        ax.text(
            x0 + M_OFF,
            label_y - 0.04,
            "M",
            ha="center",
            va="top",
            fontsize=8,
            color="steelblue",
            transform=ax.get_xaxis_transform(),
        )
    ax.text(
        nm_x,
        label_y - 0.04,
        "NM",
        ha="center",
        va="top",
        fontsize=8,
        color="darkorange",
        transform=ax.get_xaxis_transform(),
    )

ax.set_ylabel("Normalized Contamination Score", fontsize=11)
ax.set_ylim(-0.02, 0.85)
ax.set_xlim(-0.1, len(DATASETS) * BLOCK + 0.1)
ax.set_xticks([])
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Model legend (left)
model_handles = [
    plt.scatter([], [], marker=MARKERS[j], color=COLORS[j], s=80, label=MODEL_LABELS[j])
    for j in range(len(MODELS))
]
legend_models = ax.legend(
    handles=model_handles,
    title="Model",
    loc="upper left",
    framealpha=0.8,
    fontsize=9,
    title_fontsize=9,
)
ax.add_artist(legend_models)

# Member/Non-member legend (right)
member_patch = mpatches.Patch(facecolor="grey", label="Member (filled)")
nonmember_patch = mpatches.Patch(
    facecolor="white", edgecolor="grey", linewidth=1.5, label="Non-Member (hollow)"
)
ax.legend(
    handles=[member_patch, nonmember_patch],
    loc="upper right",
    framealpha=0.8,
    fontsize=9,
)

plt.tight_layout()
out = "figures/codec_scores.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")

# --- Box plot: member vs non-member distributions per model ---

fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.set_title(
    "CoDeC Normalized Scores — Member vs Non-Member per Model",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

gap = 0.3  # gap between the two boxes within a model group
spacing = 1.5  # distance between model group centres

member_color = "#e74c3c"  # red
nonmember_color = "#2980b9"  # blue

box_positions_m = []
box_positions_nm = []
box_data_m = []
box_data_nm = []
tick_positions = []
tick_labels = []

for i, model in enumerate(MODELS):
    centre = i * spacing
    pos_m = centre - gap / 2
    pos_nm = centre + gap / 2

    m_scores = [v for ds, (m, nm) in DATA[model].items() if m is not None for v in [m]]
    nm_scores = [
        v for ds, (m, nm) in DATA[model].items() if nm is not None for v in [nm]
    ]

    box_positions_m.append(pos_m)
    box_positions_nm.append(pos_nm)
    box_data_m.append(m_scores)
    box_data_nm.append(nm_scores)
    tick_positions.append(centre)
    tick_labels.append(MODEL_LABELS[i])


def style_boxes(bp, color):
    for element in ("boxes", "whiskers", "caps", "medians", "fliers"):
        plt.setp(bp[element], color=color)
    for patch in bp.get("boxes", []):
        patch.set(facecolor=color, alpha=0.35)
    plt.setp(bp["medians"], color=color, linewidth=2)
    plt.setp(
        bp.get("fliers", []),
        marker="o",
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=5,
    )


bp_m = ax2.boxplot(
    box_data_m,
    positions=box_positions_m,
    widths=0.22,
    patch_artist=True,
    notch=False,
    manage_ticks=False,
)
bp_nm = ax2.boxplot(
    box_data_nm,
    positions=box_positions_nm,
    widths=0.22,
    patch_artist=True,
    notch=False,
    manage_ticks=False,
)

style_boxes(bp_m, member_color)
style_boxes(bp_nm, nonmember_color)

ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels, fontsize=10)
ax2.set_ylabel("Normalized Contamination Score", fontsize=11)
ax2.set_xlim(-spacing / 2, (len(MODELS) - 1) * spacing + spacing / 2)
ax2.set_ylim(-0.02, 0.88)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
ax2.set_axisbelow(True)

member_patch = mpatches.Patch(
    facecolor=member_color, alpha=0.5, label="Member (3 datasets)"
)
nonmember_patch = mpatches.Patch(
    facecolor=nonmember_color, alpha=0.5, label="Non-Member (5 datasets)"
)
ax2.legend(handles=[member_patch, nonmember_patch], fontsize=10, framealpha=0.85)

plt.tight_layout()
out2 = "figures/codec_boxplot.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved {out2}")
