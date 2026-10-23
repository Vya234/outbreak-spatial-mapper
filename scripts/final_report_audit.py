from pathlib import Path
import pandas as pd
import re

out = Path("research_outputs")
report = (out / "BTP_report_draft.md").read_text()

print("FIGURE CHECK")
for name in [
    "mh_add_spatiotemporal_maps.png",
    "mh_add_reporting_aware_timeline.png",
    "mh_add_lisa_map.png",
    "mh_add_fair_model_comparison.png",
]:
    print(name, (out / name).exists())

print("\nFORECAST CHECK")
f = pd.read_csv(out / "rolling_origin_mh_add_fair_results.csv")
for cutoff, g in f.groupby("cutoff", sort=True):
    print(cutoff, "models:", len(g), "test rows:", g["test_n"].unique())

print("\nMORAN CHECK")
m = pd.read_csv(out / "mh_add_moran_summary_robustness.csv")
print(m[["summary_statistic", "moran_I", "permutation_p"]].to_string(index=False))

print("\nREPORT CHECK")
print("Word count:", len(report.split()))
print("Figure references:", len(re.findall(r'!\[.*?\]\(.*?\)', report)))
print("Conclusion present:", "## 10. Conclusion" in report)
