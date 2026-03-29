"""Generate RMSE and time heatmap plots from experiment results."""

import json
import numpy as np
import matplotlib.pyplot as plt


def load_results(path="svt_results.json"):
    with open(path) as f:
        return json.load(f)


def make_table(data, ranks, fracs, metric="rmse"):
    table = np.zeros((len(ranks), len(fracs)))
    for i, r in enumerate(ranks):
        for j, f in enumerate(fracs):
            table[i, j] = data[f"r{r}_f{f}"][metric]
    return table


def plot_heatmap(table, ranks, fracs, title, cbar_label, filename, fmt=".4f"):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(table, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([str(f) for f in fracs])
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([str(r) for r in ranks])
    ax.set_xlabel("Observation fraction f")
    ax.set_ylabel("Rank r")
    ax.set_title(title)
    for i in range(len(ranks)):
        for j in range(len(fracs)):
            val = table[i, j]
            color = "white" if val > (table.max() + table.min()) / 2 else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    color=color, fontsize=10)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def main():
    results = load_results()

    for tag in sorted(results):
        info = results[tag]
        R, F, D = info["ranks"], info["fracs"], info["data"]

        rmse_table = make_table(D, R, F, "rmse")
        time_table = make_table(D, R, F, "time")

        plot_heatmap(rmse_table, R, F,
                     f"RMSE ({tag})", "RMSE",
                     f"rmse_{tag}.png")
        plot_heatmap(time_table, R, F,
                     f"Execution Time ({tag})", "Time (s)",
                     f"time_{tag}.png", fmt=".1f")


if __name__ == "__main__":
    main()
