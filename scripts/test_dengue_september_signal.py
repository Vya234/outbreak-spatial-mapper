import pandas as pd
import numpy as np
from scipy.stats import binomtest, wilcoxon

FILE = "research_outputs/mh_dengue_september_district_year_comparison.csv"
df = pd.read_csv(FILE)

d = df["difference"].dropna().to_numpy(dtype=float)
n = len(d)

print("Eligible district-year comparisons:", n)
print("Positive:", int((d > 0).sum()))
print("Negative:", int((d < 0).sum()))
print("Tied:", int((d == 0).sum()))
print("Median difference:", float(np.median(d)))
print("Mean difference:", float(np.mean(d)))

# Exact two-sided sign test, excluding ties.
nonzero = d[d != 0]
positive = int((nonzero > 0).sum())
sign_result = binomtest(
    positive,
    n=len(nonzero),
    p=0.5,
    alternative="two-sided",
)
print("\n=== SIGN TEST ===")
print("Non-tied comparisons:", len(nonzero))
print("Positive fraction:", positive / len(nonzero))
print("Two-sided p-value:", sign_result.pvalue)

# Wilcoxon signed-rank test, excluding zero differences.
print("\n=== WILCOXON SIGNED-RANK TEST ===")
try:
    result = wilcoxon(d, alternative="two-sided", method="auto")
    print("Statistic:", result.statistic)
    print("Two-sided p-value:", result.pvalue)
except ValueError as e:
    print("Test could not be calculated:", e)

# Bootstrap confidence interval for the median paired difference.
# This resamples district-year rows, so it does not account for
# dependence between different years from the same district.
rng = np.random.default_rng(42)
B = 20000
boot = np.empty(B)

for i in range(B):
    sample = rng.choice(d, size=n, replace=True)
    boot[i] = np.median(sample)

lo, hi = np.percentile(boot, [2.5, 97.5])
print("\n=== BOOTSTRAP MEDIAN DIFFERENCE ===")
print("Median difference:", np.median(d))
print("95% bootstrap interval:", (float(lo), float(hi)))

print("\nInterpretation caution:")
print("These are exploratory tests on sparse, irregularly observed reports.")
print("They do not establish causal seasonality or statewide incidence peaks.")
