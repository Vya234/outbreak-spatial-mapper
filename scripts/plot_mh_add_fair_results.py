import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("research_outputs")
df = pd.read_csv(out / "rolling_origin_mh_add_fair_results.csv")

models = [
    "Training median",
    "District + month",
    "District + month + climate",
]
cutoffs = sorted(df["cutoff"].unique())

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, metric, title in [
    (axes[0], "MAE", "Mean Absolute Error (MAE)"),
    (axes[1], "RMSE", "Root Mean Squared Error (RMSE)"),
]:
    for model in models:
        part = df[df["model"] == model].set_index("cutoff").loc[cutoffs]
        ax.plot(
            range(len(cutoffs)), part[metric],
            marker="o", linewidth=2, label=model
        )
    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels(cutoffs, rotation=25, ha="right")
    ax.set_ylabel("Cases")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

axes[0].legend(fontsize=8)
fig.suptitle(
    "Chronological Evaluation of Maharashtra ADD Case-Count Models",
    fontsize=13
)
fig.text(
    0.5, 0.01,
    "Lower is better. Test rows are climate-complete and matched within each cutoff. "
    "Climate inputs are measured on target dates.",
    ha="center", fontsize=8
)
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(out / "mh_add_fair_model_comparison.png", dpi=300, bbox_inches="tight")
print("Saved research_outputs/mh_add_fair_model_comparison.png")
