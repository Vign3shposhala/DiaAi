"""
Gauge chart renderer for DiaForecaster AI.
Uses a proper filled semicircle arc approach — no polar projection quirks.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

RISK_COLORS = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444', 'Critical': '#7c3aed'}

ZONES = [
    (0,    0.25, '#10b981', 'Low'),
    (0.25, 0.50, '#f59e0b', 'Medium'),
    (0.50, 0.75, '#ef4444', 'High'),
    (0.75, 1.00, '#7c3aed', 'Critical'),
]


def draw_gauge(prob: float, level: str, figsize=(5, 3.2)) -> plt.Figure:
    """
    Draw a clean semicircle gauge showing risk probability.

    Args:
        prob:    float in [0, 1]
        level:   'Low' | 'Medium' | 'High' | 'Critical'
        figsize: matplotlib figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.30, 1.25)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── Draw arc zones ──
    n_pts = 300
    r_inner, r_outer = 0.58, 1.0
    ring_width = np.linspace(r_inner, r_outer, 28)

    for p_start, p_end, color, _ in ZONES:
        a_start = np.pi * (1 - p_start)   # 180° → 0°  maps 0→1
        a_end   = np.pi * (1 - p_end)
        angles  = np.linspace(a_start, a_end, n_pts)

        # Outer arc
        x_out = r_outer * np.cos(angles)
        y_out = r_outer * np.sin(angles)
        # Inner arc (reversed for fill)
        x_in  = r_inner * np.cos(angles[::-1])
        y_in  = r_inner * np.sin(angles[::-1])

        xs = np.concatenate([x_out, x_in])
        ys = np.concatenate([y_out, y_in])
        ax.fill(xs, ys, color=color, alpha=0.92, zorder=2)

        # Zone label
        a_mid = (a_start + a_end) / 2
        r_lbl = (r_inner + r_outer) / 2
        lbl_x = (r_outer + 0.16) * np.cos(a_mid)
        lbl_y = (r_outer + 0.16) * np.sin(a_mid)
        ax.text(lbl_x, lbl_y, _,
                ha='center', va='center', fontsize=8.5,
                fontweight='700', color=color,
                fontfamily='DejaVu Sans')

    # ── Thin separator lines between zones ──
    for p in [0.25, 0.50, 0.75]:
        angle = np.pi * (1 - p)
        ax.plot([r_inner * np.cos(angle), r_outer * np.cos(angle)],
                [r_inner * np.sin(angle), r_outer * np.sin(angle)],
                color='white', linewidth=2.5, zorder=3)

    # ── Background arc (track) ──
    bg_angles = np.linspace(0, np.pi, 300)
    for r in np.linspace(r_inner, r_outer, 8):
        ax.plot(r * np.cos(bg_angles), r * np.sin(bg_angles),
                color='#f1f5f9', linewidth=0.5, zorder=1)

    # ── Needle ──
    needle_angle = np.pi * (1 - np.clip(prob, 0.01, 0.99))
    needle_len   = 0.82
    nx = needle_len * np.cos(needle_angle)
    ny = needle_len * np.sin(needle_angle)

    # Needle shadow
    ax.annotate("",
                xy=(nx * 1.01, ny * 1.01),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color='#94a3b8',
                    lw=3.5,
                    mutation_scale=14,
                ),
                zorder=4)
    # Needle body
    ax.annotate("",
                xy=(nx, ny),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color='#0f172a',
                    lw=2.5,
                    mutation_scale=13,
                ),
                zorder=5)

    # ── Hub circle ──
    hub = plt.Circle((0, 0), 0.07, color='#0f172a', zorder=6)
    ax.add_patch(hub)
    hub_ring = plt.Circle((0, 0), 0.10, color='white', zorder=5)
    ax.add_patch(hub_ring)
    hub2 = plt.Circle((0, 0), 0.07, color='#0f172a', zorder=6)
    ax.add_patch(hub2)

    # ── Centre text ──
    risk_color = RISK_COLORS[level]
    ax.text(0, -0.17, f"{prob*100:.1f}%",
            ha='center', va='center',
            fontsize=18, fontweight='700',
            color=risk_color, zorder=7,
            fontfamily='DejaVu Sans')
    ax.text(0, -0.28, level.upper(),
            ha='center', va='center',
            fontsize=9.5, fontweight='600',
            color='#64748b', zorder=7,
            fontfamily='DejaVu Sans')

    # ── Min / Max labels ──
    ax.text(-1.08, -0.08, '0%',   ha='center', va='center',
            fontsize=8, color='#94a3b8', fontweight='500')
    ax.text( 1.08, -0.08, '100%', ha='center', va='center',
            fontsize=8, color='#94a3b8', fontweight='500')

    plt.tight_layout(pad=0.2)
    return fig
