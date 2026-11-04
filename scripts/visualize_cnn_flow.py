"""
Generate an animated GIF showing data flow through EEGNet, ShallowFBCSPNet,
Deep4Net, EEGTCNet, and EEGConformer.
Output: results/cnn_data_flow.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Layer definitions from traced shapes ─────────────────────────────────────
# Shape tuples use (F, T, C_spatial) convention for 3-D tensors,
# (F, T) for 2-D tensors, and (1,) for scalars.

MODELS = {
    "EEGNet": {
        "color": "#4C9BE8",
        "layers": [
            ("Input",                       (129, 200),     "#778899"),
            ("Temporal Conv\nk=(1,64)",     (8, 200, 129),  "#4C9BE8"),
            ("Depthwise Spatial\nk=(129,1)",(16, 1, 201),   "#2E75B6"),
            ("ELU + Pool\n÷4 time",         (16, 1, 50),    "#1A5C94"),
            ("Separable Conv\nk=(1,16)",    (16, 1, 51),    "#2E75B6"),
            ("ELU + Pool\n÷8 time",         (16, 1, 6),     "#1A5C94"),
            ("Linear",                      (1,),           "#0D2F4F"),
            ("RT Prediction\n(scalar)",     (1,),           "#FF6B35"),
        ]
    },
    "ShallowFBCSPNet": {
        "color": "#52B788",
        "layers": [
            ("Input",                           (129, 200),     "#778899"),
            ("Temporal Conv\nk=25, F=40",       (40, 176, 129), "#52B788"),
            ("Spatial Conv\nk=(1,129)",         (40, 176, 1),   "#2D9B5A"),
            ("BN + Square\n(power)",            (40, 176, 1),   "#1E7A44"),
            ("AvgPool + Log\n÷25 stride=15",    (40, 7, 1),     "#2D9B5A"),
            ("Dropout",                         (40, 7, 1),     "#1E7A44"),
            ("Linear",                          (1,),           "#0D4A28"),
            ("RT Prediction\n(scalar)",         (1,),           "#FF6B35"),
        ]
    },
    "Deep4Net": {
        "color": "#E07B54",
        "layers": [
            ("Input",                       (129, 200),     "#778899"),
            ("Conv Block 1\nF=25, k=10",    (25, 191, 129), "#E07B54"),
            ("Conv Block 2\nF=50, k=10",    (50, 188, 1),   "#C45E3A"),
            ("Conv Block 3\nF=100, k=10",   (100, 185, 1),  "#A84522"),
            ("Conv Block 4\nF=200, k=10",   (200, 182, 1),  "#8C2C0A"),
            ("Global Pool",                 (200, 1),        "#C45E3A"),
            ("Linear",                      (1,),           "#5C1500"),
            ("RT Prediction\n(scalar)",     (1,),           "#FF6B35"),
        ]
    },
    "EEGTCNet": {
        "color": "#9B59B6",
        "layers": [
            ("Input",                           (129, 200),     "#778899"),
            ("Temporal Conv\nF1=8, k=64",       (8, 200, 129),  "#9B59B6"),
            ("Depthwise Spatial\nk=(129,1)",    (16, 200, 1),   "#8E44AD"),
            ("ELU + Pool\n÷8 time",             (16, 25, 1),    "#7D3C98"),
            ("Separable Conv\n+ Pool ÷8",       (16, 3, 1),     "#6C3483"),
            ("TCN Blocks\ndepth=2, F=12",       (12, 3),        "#5B2C6F"),
            ("Linear",                          (1,),           "#4A235A"),
            ("RT Prediction\n(scalar)",         (1,),           "#FF6B35"),
        ]
    },
    "EEGConformer": {
        "color": "#16A085",
        "layers": [
            ("Input",                               (129, 200),     "#778899"),
            ("Temporal Conv\nn_filters=40, k=25",  (40, 176, 129), "#1ABC9C"),
            ("Spatial Conv\nk=(129,1)",             (40, 176, 1),   "#16A085"),
            ("BN + ELU\n+ Pool ÷15",               (40, 7, 1),     "#148F77"),
            ("Patch Embed\n7 tokens × 40d",         (40, 7),        "#117A65"),
            ("Transformer Enc\n6× MHA (10 heads)", (40, 7),        "#0E6655"),
            ("MLP Head\n280 → 256 → 32",           (1,),           "#0B5345"),
            ("RT Prediction\n(scalar)",             (1,),           "#FF6B35"),
        ]
    },
}


def shape_to_label(shape):
    """Human-readable tensor shape."""
    if len(shape) == 3:
        return f"{shape[0]}ch × {shape[1]}t × {shape[2]}"
    elif len(shape) == 2:
        return f"{shape[0]}ch × {shape[1]}t"
    else:
        return "scalar"


def shape_to_size(shape):
    """Map tensor volume to a display height (capped for readability)."""
    if len(shape) == 1:
        return 0.15
    vol = 1
    for s in shape:
        vol *= s
    return min(0.08 + 0.35 * (np.log10(vol + 1) / 5), 0.75)


# ── Figure layout (2 rows: 3 models top, 2 centred bottom) ───────────────────

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0F1923")
gs = GridSpec(2, 6, figure=fig, hspace=0.12, wspace=0.08)

axes = [
    fig.add_subplot(gs[0, 0:2]),  # EEGNet
    fig.add_subplot(gs[0, 2:4]),  # ShallowFBCSPNet
    fig.add_subplot(gs[0, 4:6]),  # Deep4Net
    fig.add_subplot(gs[1, 1:3]),  # EEGTCNet
    fig.add_subplot(gs[1, 3:5]),  # EEGConformer
]

N_FRAMES_PER_MODEL = 28
N_MODELS = 5
TOTAL_FRAMES = N_FRAMES_PER_MODEL * N_MODELS + 15  # +15 for final hold


def draw_model(ax, model_name, model_data, active_layer_idx, alpha_layers):
    ax.clear()
    ax.set_facecolor("#0F1923")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    layers = model_data["layers"]
    n = len(layers)
    base_color = model_data["color"]

    # Title
    ax.text(0.5, 0.97, model_name, ha="center", va="top",
            fontsize=13, fontweight="bold", color=base_color,
            transform=ax.transAxes)

    # Evenly spaced y-positions from top to bottom
    y_positions = np.linspace(0.90, 0.05, n)
    box_w = 0.80
    box_x = 0.10

    for i, (layer_name, shape, color) in enumerate(layers):
        y = y_positions[i]
        h = shape_to_size(shape) * 0.10 + 0.045

        alpha = alpha_layers[i]
        is_active = (i == active_layer_idx)

        # Box
        fancy = FancyBboxPatch(
            (box_x, y - h / 2), box_w, h,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor="white" if is_active else color,
            linewidth=2.5 if is_active else 0.5,
            alpha=alpha,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(fancy)

        # Glow on active
        if is_active:
            glow = FancyBboxPatch(
                (box_x - 0.02, y - h / 2 - 0.008), box_w + 0.04, h + 0.016,
                boxstyle="round,pad=0.01",
                facecolor="none",
                edgecolor="white",
                linewidth=1.0,
                alpha=0.3 * alpha,
                transform=ax.transAxes,
                zorder=1,
            )
            ax.add_patch(glow)

        # Layer name (left-aligned)
        ax.text(box_x + 0.03, y, layer_name,
                ha="left", va="center",
                fontsize=8, color="white", alpha=alpha,
                fontweight="bold" if is_active else "normal",
                transform=ax.transAxes, zorder=3)

        # Shape label (right-aligned)
        shape_str = shape_to_label(shape)
        ax.text(box_x + box_w - 0.03, y, shape_str,
                ha="right", va="center",
                fontsize=7, color="#CCDDEE", alpha=alpha,
                transform=ax.transAxes, zorder=3)

        # Arrow to next layer
        if i < n - 1:
            y_next = y_positions[i + 1]
            arr_alpha = min(alpha_layers[i], alpha_layers[i + 1])
            ax.annotate(
                "", xy=(0.5, y_next + shape_to_size(layers[i + 1][1]) * 0.05 + 0.03),
                xytext=(0.5, y - h / 2 - 0.005),
                xycoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#AABBCC",
                    lw=1.2,
                    alpha=arr_alpha * 0.7,
                ),
                zorder=1,
            )

    # Pulse label on active layer
    if 0 <= active_layer_idx < n:
        y_a = y_positions[active_layer_idx]
        ax.text(0.5, y_a - 0.07, "◀ processing ▶",
                ha="center", va="center", fontsize=7,
                color=base_color, alpha=0.8,
                transform=ax.transAxes)


def make_alpha_layers(model_idx, frame_in_model, n_layers):
    """Per-layer alpha values and the active layer index."""
    progress = frame_in_model / N_FRAMES_PER_MODEL  # 0..1
    active = int(progress * (n_layers - 1))
    active = min(active, n_layers - 1)

    alphas = []
    for i in range(n_layers):
        if i < active:
            alphas.append(0.75)
        elif i == active:
            sub = (progress * (n_layers - 1)) - active
            alphas.append(0.35 + 0.65 * sub)
        else:
            alphas.append(0.18)
    return alphas, active


def animate(frame):
    model_names = list(MODELS.keys())

    for m_idx, (ax, name) in enumerate(zip(axes, model_names)):
        model_data = MODELS[name]
        n_layers = len(model_data["layers"])

        if frame < m_idx * N_FRAMES_PER_MODEL:
            alphas = [0.18] * n_layers
            active = -1
        elif frame >= (m_idx + 1) * N_FRAMES_PER_MODEL:
            alphas = [0.85] * n_layers
            active = n_layers
        else:
            frame_in_model = frame - m_idx * N_FRAMES_PER_MODEL
            alphas, active = make_alpha_layers(m_idx, frame_in_model, n_layers)

        draw_model(ax, name, model_data, active, alphas)

    fig.suptitle(
        "EEG Model Data Flow  —  Input (129 ch × 200 samples)  →  RT prediction (scalar)",
        fontsize=13, color="#AACCEE", y=0.998, fontweight="bold",
        fontfamily="monospace",
    )


# ── Render ────────────────────────────────────────────────────────────────────

out_path = Path("results/cnn_data_flow.gif")
out_path.parent.mkdir(parents=True, exist_ok=True)

ani = FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=80, blit=False)

print(f"Rendering {TOTAL_FRAMES} frames …")
ani.save(str(out_path), writer=PillowWriter(fps=12), dpi=80)
print(f"Saved → {out_path}")
plt.close()
