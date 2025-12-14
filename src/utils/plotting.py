import pandas as pd
import matplotlib.pyplot as plt


def plot_map(static_path):

    stops = pd.read_csv(static_path / 'stops.txt', low_memory=False)
    shapes = pd.read_csv(static_path / 'shapes.txt', low_memory=False)

    fig, ax = plt.subplots(1, figsize=(16,9), dpi=450, facecolor='black')
    ax.set_facecolor('black')

    ax.scatter(stops['stop_lon'], stops['stop_lat'], s=2, c='white', alpha=0.7)
    for _, group in shapes.groupby("shape_id"):
        group = group.sort_values("shape_pt_sequence")
        ax.plot(group["shape_pt_lon"], group["shape_pt_lat"], linewidth=0.3, c='white')

    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig('graph.png', facecolor=fig.get_facecolor())
    plt.close(fig)