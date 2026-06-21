"""
visualize.py
------------
Renders the 3D PCA-reduced, K-Means-clustered embeddings as a minimal,
dark-themed 3D scatter plot using square ("cube-like") markers.

This intentionally uses matplotlib's standard `Axes3D.scatter` with
`marker="s"` rather than hand-built 3D cube geometry -- it's the typical,
idiomatic way to swap point shape in the standard data-science plotting
stack (matplotlib / seaborn / plotly all expose marker style as a simple
argument), and it keeps the plot flat and minimal like a normal scatter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Blue / grey / black palette (dark theme, distinct from the spherical
# reference image's purple/teal/orange scheme)
PALETTE = {
    "ml_learning_booklets": "#4FA8F5",                   # bright blue
    "nlp_learning_booklets": "#7C8B9B",                  # mid steel grey
    "data_science_learning_booklets": "#2E4A6B",         # deep navy
}

BG_COLOR = "#0E1117"
PANEL_COLOR = "#11151C"
GRID_COLOR = (0.5, 0.55, 0.6, 0.18)


def plot_cube_clusters(df, out_path="output/cluster_plot.png", marker_size=90):
    fig = plt.figure(figsize=(9, 7), facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    # Minimal panes: no edge lines, panel fill blends into the background,
    # and only a faint grid remains.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANEL_COLOR)
        axis.pane.set_alpha(1.0)
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis.line.set_color((0, 0, 0, 0))  # hide the bright axis frame lines

    for topic, color in PALETTE.items():
        sub = df[df["true_topic"] == topic]
        ax.scatter(
            sub["pc1"], sub["pc2"], sub["pc3"],
            marker="s", s=marker_size, c=color,
            edgecolors="none", linewidths=0, alpha=0.9,
            depthshade=True, label=topic,
        )

    ax.set_xlabel("PC1", color="#9AA5B1")
    ax.set_ylabel("PC2", color="#9AA5B1")
    ax.set_zlabel("PC3", color="#9AA5B1")
    ax.tick_params(colors="#6B7280")
    ax.set_title("Clustering with synthetic document embeddings", color="#E5E9EE", loc="left", fontsize=11)

    legend = ax.legend(
        loc="lower left", bbox_to_anchor=(0.0, -0.05), facecolor=PANEL_COLOR,
        edgecolor="none", fontsize=8, labelcolor="#C7D3DC", ncol=2,
        title="label", title_fontsize=8,
    )
    legend.get_title().set_color("#9AA5B1")

    ax.view_init(elev=18, azim=-60)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor=BG_COLOR)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    df = pd.read_csv("data/clustered_embeddings.csv")
    plot_cube_clusters(df)
