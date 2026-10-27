### 6.6 Outbreak-occurrence forecasting (district-month panel)

**Motivation.** EpiClim rows are reported outbreak events. A district-month without a row means that no report was recorded, not that zero cases occurred (Section 3.3). The case-count experiments in Sections 6.1–6.5 condition on a report existing. This experiment instead asks whether a district will report an outbreak of a given disease in the next month.

**Problem definition.** For each district and month *m*, the target is whether the district records at least one reported outbreak of the disease in month *m*+1. Inputs use only information from months up to *m*. The panel covers 795 districts (taken from the full dataset) over 158 months (May 2009–June 2022). The target diseases are Acute Diarrhoeal Disease (5,126 records) and Dengue (1,619 records), matched on the exact `Disease` label. Related labels are excluded: 79 records containing "dengue" (for example "Dengue Fever", "Suspected Dengue" and mixed dengue/chikungunya labels) and 102 gastroenteritis and diarrhoea variants. Across the test years, 3.5% of district-months had a reported ADD outbreak and 1.4% had a reported dengue outbreak.

**Features.** All features are computed from months up to *m*:

- Season: sine and cosine of the target calendar month.
- Own district: reports of the target disease in the last month, last 3 months and last 12 months, and any-disease reports in the last 3 months.
- Rest of the state: reports of the target disease in month *m*, and any-disease reports in the last 3 months, in the other districts of the same state.
- Neighbouring districts: the mean number of reports in the last month and last 3 months across the five nearest district centroids. District coordinates are fixed per district in the dataset. This differs from the Queen-contiguity weights used for Maharashtra in Section 5, because no national boundary file is used here.

**Models.** The baseline ranks districts by their number of reports of the target disease in the previous 12 months. Three gradient-boosted tree models (scikit-learn `HistGradientBoostingClassifier`; maximum depth 4, learning rate 0.05, 200 iterations, random seed 0, no hyperparameter tuning) use cumulative feature sets: season and state activity; plus own-district history; plus neighbouring districts.

**Evaluation.** Rolling-origin: for each test year from 2016 to 2019, models are trained on all earlier months and tested on the 12 target months of that year. The years 2020–2022 are excluded because reporting drops sharply (151, 367 and 188 records, compared with 572–1,113 per year in 2010–2019). Metrics are PR-AUC (chance level equals the base rate), ROC-AUC, and recall in the top 20 districts per month. For the last metric, in each test month with at least one reported outbreak, the 795 districts are ranked by predicted score and the fraction of that month's outbreak districts found among the top 20 is recorded (chance level 20/795 = 2.5%).

**Results.** Table 6.6 gives the means over the four test years.

| Disease | Model | PR-AUC | ROC-AUC | Recall in top 20 |
|---|---|---:|---:|---:|
| Acute Diarrhoeal Disease (base rate 0.035) | Own 12-month count (baseline) | 0.085 | 0.680 | 0.120 |
| | GBM: season + state | 0.077 | 0.671 | 0.051 |
| | GBM: + own-district history | 0.129 | 0.755 | 0.133 |
| | GBM: + neighbouring districts | 0.131 | 0.756 | 0.130 |
| Dengue (base rate 0.014) | Own 12-month count (baseline) | 0.065 | 0.705 | 0.180 |
| | GBM: season + state | 0.056 | 0.749 | 0.171 |
| | GBM: + own-district history | 0.126 | 0.819 | 0.239 |
| | GBM: + neighbouring districts | 0.125 | 0.822 | 0.242 |

*Table 6.6. Mean over test years 2016–2019. Source: `research_outputs/occurrence_rolling_results.csv`. Figure 5 (Section 11.5) shows the same results with variation across years.*

- Models that use the district's own history are well above chance (PR-AUC 0.129 against a base rate of 0.035 for ADD; 0.126 against 0.014 for dengue).
- Compared with the baseline, the model with own-district history had higher PR-AUC in all four test years for both diseases. For recall in the top 20 districts it was higher in all four years for dengue and in three of four years for ADD (not in 2019).
- The variation across test years is large (error bars in Figure 5), so the size of the improvement over the baseline is uncertain. For ADD, the top-20 recall gain is small (0.133 against 0.120).
- Season and state activity alone did not beat the baseline on PR-AUC (ADD 0.077 against 0.085; dengue 0.056 against 0.065), so a district's own reporting history carries most of the signal.
- Neighbouring-district features gave no measurable gain (PR-AUC 0.131 against 0.129 for ADD; 0.125 against 0.126 for dengue). This is consistent with the non-significant Moran's I in Section 5, but it does not show that spatial transmission is absent: it only shows that, under this neighbour definition and these data, nearby reports did not improve next-month prediction.

**Robustness to reporting footprint.** Unreported months cannot be confirmed as true negatives, so the experiment was repeated on districts with a stronger reporting history. For each test year, a district was eligible if it had at least 6 (or at least 12) months with a recorded report of the target disease in the training period only. Models and features were unchanged, and only the test rows were filtered (`occurrence_robustness.py`). The thresholds are exploratory choices. Table 6.7 gives the results.

| Disease | Eligible districts | Positives per year | Base rate | Baseline PR-AUC | GBM (own history) PR-AUC | GBM lift over base rate |
|---|---|---:|---:|---:|---:|---:|
| ADD, all districts | 795 | 273–477 | 0.035 | 0.085 | 0.129 | 3.7× |
| ADD, ≥6 recorded months | 152–228 | 144–226 | 0.077 | 0.127 | 0.180 | 2.4× |
| ADD, ≥12 recorded months | 33–78 | 59–84 | 0.112 | 0.167 | 0.231 | 2.1× |
| Dengue, all districts | 795 | 97–152 | 0.014 | 0.065 | 0.126 | 9.2× |
| Dengue, ≥6 recorded months (low sample) | 26–48 | 26–49 | 0.083 | 0.125 | 0.246 | 2.9× |
| Dengue, ≥12 recorded months (low sample) | 5–15 | 8–27 | 0.129 | 0.172 | 0.281 | 2.2× |

*Table 6.7. Mean over test years 2016–2019. PR-AUC is not comparable across rows because base rates differ. "Low sample" marks sets with fewer than 30 positive district-months in at least one test year; this cut-off is a practical warning rule, not a statistical standard. Source: `research_outputs/occurrence_robustness_summary.csv`.*

- The model with own-district history had higher PR-AUC than the 12-month-count baseline in all four test years in every eligibility set, for both diseases.
- Its lift over the base rate shrinks as eligibility becomes stricter (ADD 3.7×, 2.4×, 2.1×; dengue 9.2×, 2.9×, 2.2×). Part of the performance on all districts therefore comes from separating districts that report at all from those that rarely do. Among districts that already report regularly, the model still ranks future reports above chance, but by a smaller margin.
- The restricted dengue sets contain only 5–48 districts and 8–49 positives per year, so those estimates are exploratory.
- Neighbouring-district features again gave no consistent gain (for example ADD ≥6: 0.180 against 0.180; ADD ≥12: 0.227 against 0.231).
- A district with many past reports is not necessarily fully monitored, so this analysis does not resolve the missing-negative-label problem.

**Permutation importance.** Figure 6 (Section 11.5) shows the drop in PR-AUC on the 2019 test year when each feature is shuffled. For ADD, the largest values were the own-district 12-month count (0.0298), the seasonal cosine term (0.0122) and other districts in the state reporting in the last month (0.0052); all others were 0.003 or smaller. For dengue, the other-districts-in-state feature (0.0572, SD 0.0077) and the own-district 12-month count (0.0548, SD 0.0048) were the largest and are not clearly separable, followed by season (cosine 0.0159) and the own-district last-month report (0.0123). The neighbouring-district features had negative values for dengue (−0.0079 and −0.0008), which indicates that they contribute no useful signal.

**Interpretation.** These importance values describe which inputs the model uses to predict *reported* occurrence. A district's own history partly reflects how often it reports at all, so the values must not be read as causes of disease spread. They are computed on a single test year, and correlated features share importance.

**What this experiment does not show.**

- It evaluates whether a district reports an outbreak next month. It does not validate the seven-day geographic expansion overlay, which remains unvalidated.
- Outcomes are reports, not true incidence.
- The district list and coordinates come from the full dataset, including later years, which is a minor source of leakage in the definition of the district universe.
- Only one hyperparameter setting was used, and the four test years share training data and are not independent.
- Climate variables were not used: they are present only on reported rows (5,997 of 8,985 rows have all three), so no climate values exist for district-months without a report. Lagged climate was therefore not evaluated.
- Only two diseases were studied, and the years 2020–2022 were not evaluated.

**Reproducibility.** `occurrence_model.py` writes `occurrence_rolling_results.csv` and `occurrence_permutation_importance.csv` to `research_outputs/`, `plot_occurrence_results.py` produces Figures 5 and 6 from those files, and `occurrence_robustness.py` writes `occurrence_robustness_by_year.csv` and `occurrence_robustness_summary.csv`. Results were produced with scikit-learn 1.7.2 and pandas 2.3.3.