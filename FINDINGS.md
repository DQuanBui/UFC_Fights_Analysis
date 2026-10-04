# Research findings

Main cohort: UFC events, 1994–8 August 2026. All values are recomputed from the supplied snapshot.

## 1. Scope is an analytical decision

**Evidence.** The main cohort contains 8,820 UFC fights across 783 events from 1994 through 8 August 2026; the raw source contains 11,441 fights across multiple promotions.

**Interpretation.** UFC conclusions require explicit event classification. Eight 1993 UFC fights are retained outside the requested window and supply earlier history.

Source tables: annual.csv; source_summary.csv

## 2. Event volume expanded, then stabilized

**Evidence.** The snapshot has 3 events in 1994, 24 in 2010 and 42 in 2025.

**Interpretation.** Recent changes in performance should not automatically be explained by continued event-count growth. Do not annualize the partial 2026 count.

Source tables: annual.csv

## 3. A small majority of fights end by KO/TKO or submission

**Evidence.** Finishes account for 52.2% of all selected fights; KO/TKO alone accounts for 32.8%.

**Interpretation.** Draws, no contests, DQs and other methods remain in the denominator; the finish definition is deliberately narrow.

Source tables: annual.csv

## 4. Heavyweight finishing is concentrated in knockouts

**Evidence.** Across 784 heavyweight fights, KO/TKO accounts for 51.4%, versus 32.8% overall.

**Interpretation.** Division is a major descriptive factor. This does not isolate a causal effect of body weight.

Source tables: divisions.csv

## 5. The apparent decision trend depends on division composition

**Evidence.** Observed decision rates are 48.24% in 2010–2019 and 48.93% in 2020–2025. Fixed shared-division weights yield 48.94% and 47.82%.

**Interpretation.** The direction reverses under standardization. This is a composition diagnostic restricted to adequately represented shared divisions.

Source tables: division_mix_adjustment.csv

## 6. Finish risk declines among later-round survivors

**Evidence.** For standard three-round fights, the conditional finish rate is 26.8% in round 1, 22.2% in round 2 and 13.6% in round 3.

**Interpretation.** Each denominator is the number reaching that round. Survivor selection and changing time exposure prevent a causal fatigue interpretation.

Source tables: round_hazard.csv

## 7. Younger fighters win more often in unequal-age matchups

**Evidence.** The older fighter wins 42.9% of 8,537 decisive unequal-age bouts. Winners are 0.89 years younger on average within matched bouts.

**Interpretation.** This association can reflect selection, experience, division and matchup quality. Repeated fighter appearances remain a dependence limitation.

Source tables: advantage_uncertainty.csv; paired_statistics.csv

## 8. Reach advantage has a modest unadjusted association

**Evidence.** The longer-reach fighter wins 52.0% of 6,629 eligible fights; the event-bootstrap 95% interval is 50.8%–53.2%.

**Interpretation.** The effect is small relative to uncertainty about individual outcomes. Missing or equal reach and unresolved results are excluded.

Source tables: advantage_uncertainty.csv

## 9. Career volume and efficiency answer different questions

**Evidence.** Jim Miller leads the selected win-count table with 28 wins in 47 bouts; his decisive-fight win rate is 60.9%.

**Interpretation.** Count rankings reward longevity. Percentage rankings require at least 10 decisive fights and show their sample sizes.

Source tables: fighters.csv

## 10. Winners generate more significant-strike output

**Evidence.** Winners land 4.20 significant strikes per minute versus 2.78 for losers, using pooled observed exposure.

**Interpretation.** This is a post-fight description, not a valid predictor from the same bout. Per-minute rates and raw totals answer different questions.

Source tables: performance_by_result.csv

## 11. Grappling output also separates winners and losers

**Evidence.** Winners average 2.07 landed takedowns per 15 minutes versus 0.94 for losers.

**Interpretation.** The association combines style, opponent quality and control of the fight. It does not establish that attempting more takedowns causes a win.

Source tables: performance_by_result.csv

## 12. Title fights remain longer within five-round schedules

**Evidence.** Among standard five-round bouts, title fights average 15.95 minutes and non-title fights 14.44, a 1.52-minute gap.

**Interpretation.** Restricting scheduled length removes one obvious confounder; era, divisions and fighter selection remain. The rank-biserial effect is small.

Source tables: title_statistics.csv

## 13. Bonus categories change at the 2014 boundary

**Evidence.** Performance of the Night first appears in 2014; KO and submission categories last appear in 2014 in this snapshot.

**Interpretation.** Comparisons across time must account for category changes. No individual Performance bonus rankings are inferred from fight-only rows.

Source tables: bonus_categories.csv

## 14. Finishing bouts are more often decorated

**Evidence.** A bonus category is attached to 41.6% of KO/TKO bouts and 9.0% of decision bouts.

**Interpretation.** Rates include pre-bonus years and are descriptive. Award rules and era composition confound this difference; decorated fights are not individual payouts.

Source tables: bonus_rates.csv

## 15. International representation contracted in 2020

**Evidence.** Known-location non-USA event share falls from 47.6% in 2019 to 26.8% in 2020, then reaches 33.3% in 2025.

**Interpretation.** This timing is consistent with the COVID-era disruption, but the dataset alone cannot identify its causes. Locations are country or territory labels.

Source tables: annual.csv

## 16. Historical features provide limited predictive value

**Evidence.** Validation selected gradient boosting; test accuracy is 60.3% (event-bootstrap 95% interval 58.1%–62.6%), ROC-AUC 0.652, versus 49.5% baseline accuracy.

**Interpretation.** The model remains uncertain and is not a betting system. Test periods include previously observed fighters; this evaluates future bouts, not generalization to entirely unseen fighters.

Source tables: model_metrics.csv
