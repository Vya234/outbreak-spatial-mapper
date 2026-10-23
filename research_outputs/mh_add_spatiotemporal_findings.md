# Spatio-Temporal Analysis of Acute Diarrhoeal Disease in Maharashtra

## 1. Objective
Explore changes in district-level recorded Acute Diarrhoeal Disease (ADD)
cases over time and assess whether the observed changes support a consistent
directional pattern.

## 2. Data and methodology
The analysis uses 445 records from the prepared Maharashtra ADD dataset.
Records were grouped into three periods: 2009–2013, 2014–2018, and 2019–2022.
District medians were calculated from available observations. Districts with
no observations in a period were left unreported rather than assigned zero.

For temporal comparisons, a district was included only when it had at least
five observations in each of the two periods being compared. Sholapur was
normalized to Solapur for comparison purposes.

## 3. Reporting coverage
| Period | Districts observed | Records |
|---|---:|---:|
| 2009–2013 | 29 | 178 |
| 2014–2018 | 32 | 225 |
| 2019–2022 | 18 | 42 |

Reporting coverage is substantially lower in 2019–2022. Therefore, apparent
spatial or temporal changes may reflect changes in reporting coverage as well
as changes in recorded cases.

## 4. Period-wise district comparison
Fifteen districts met the minimum observation threshold in both 2009–2013 and
2014–2018. Eight had higher district medians and seven had lower medians.

The median of district-level changes was +6 cases, while the mean change was
−4.97 cases. The difference reflects the influence of large decreases in
some districts, including Pune and Sangli.

Examples:
- Gadchiroli: median increased from 25 to 54.
- Nanded: median increased from 21.5 to 39.
- Sangli: median decreased from 95 to 47.
- Pune: median decreased from 131 to 36.

Only Kolhapur met the five-observation threshold in both 2014–2018 and
2019–2022. This is insufficient for a meaningful statewide comparison of
those periods.

## 5. Statistical tests
For the 15 districts comparable between 2009–2013 and 2014–2018:
- Wilcoxon signed-rank test: statistic = 51.5, p = 0.629.
- Two-sided sign test: p = 1.000.

Neither test provides evidence of a consistent directional shift at the
conventional 5% significance level.

## 6. Interpretation and limitations
These results describe recorded case counts, not population-adjusted disease
incidence. The dataset is sparse and irregularly sampled, and district-level
observation counts vary across periods. The results cannot establish that
underlying disease incidence remained stable or that observed changes were
caused by changes in disease transmission.

The five-observation threshold is a practical screening rule, not a guarantee
of statistical reliability. The period comparisons are exploratory and should
not be interpreted as causal evidence.

## 7. Outputs
- `mh_add_spatiotemporal_maps.png`: period-wise district median maps using a
  shared colour scale.
- `mh_add_spatiotemporal_summary.csv`: district-level median cases and
  observation counts by period.
- `mh_add_period_change_uncertainty.csv`: medians, quartiles, and changes for
  the 15 districts comparable across the first two periods.

## 8. Recommended next step
Develop finer-grained temporal visualizations while explicitly displaying
reporting coverage. Assess whether the available sampling supports such
comparisons before making claims about trends or forecasting performance.

## 9. Monthly Temporal Analysis and Reporting Coverage

A monthly timeline was constructed from the available Maharashtra ADD records,
covering July 2009 to June 2022.

The analysis reindexed the observations to a complete monthly calendar:
- Calendar months: 156
- Months without observations: 40
- Observed months with fewer than 3 records: 55

Three panels were produced to show monthly recorded observations, the number
of districts reporting, and median recorded cases. The case-summary panel
uses a logarithmic y-axis to make lower values easier to inspect, while
marking months with fewer than three records.

The reporting series is highly irregular. The largest observed monthly median
was 607 cases in April 2020, but it came from only one record from one district.
It must not be interpreted as evidence of a statewide outbreak peak.

These results describe the available records, not continuous disease
incidence. Missing months were retained as missing values rather than
interpreted as zero cases. Apparent temporal fluctuations may reflect changes
in reporting coverage as well as changes in recorded cases.

Output files:
- `mh_add_reporting_aware_timeline.png`
- `mh_add_reporting_aware_monthly.csv`

## 10. Next Research Direction

The next stage is to strengthen the spatio-temporal forecasting component.
Before interpreting forecast performance, the evaluation must distinguish
prediction of recorded case counts from prediction of future spatial spread.
Any future model should be evaluated chronologically and compared against a
simple baseline, with reporting sparsity and missing observations documented.
