import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("research_outputs")
df = pd.read_csv(out / "rolling_origin_mh_add_fair_results_corrected.csv")

models = [
    "Training median",
    "District + month",
    "District + month + climate",
]
cutoffs = sorted(df["cutoff"].astype(str).unique())

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, metric, title in [
    (axes[0], "MAE", "Mean Absolute Error (MAE)"),
    (axes[1], "RMSE", "Root Mean Squared Error (RMSE)"),
]:
    for model in models:
        part = df[df["model"] == model].copy()
        part["cutoff"] = part["cutoff"].astype(str)
        part = part.set_index("cutoff").loc[cutoffs]

        ax.plot(
            range(len(cutoffs)),
            part[metric],
            marker="o",
            linewidth=2,
            label=model,
        )

    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels(cutoffs, rotation=25, ha="right")
    ax.set_ylabel("Cases")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

axes[0].legend(fontsize=8)
fig.suptitle(
    "Chronological Evaluation of Maharashtra ADD Case-Count Models",
    fontsize=13,
)
fig.text(
    0.5,
    0.01,
    "All eligible rows retained; missing climate features imputed. "
    "Climate inputs are measured on target dates.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=[0, 0.06, 1, 0.93])

output = out / "mh_add_fair_model_comparison_corrected.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {output}")
