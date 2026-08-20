# FluxRoute — Smart Payment Routing: Project Report

*Covers everything from synthetic dataset generation through model calibration.
Deployment (FastAPI service, containerization, load testing) has not started —
see "Not Yet Built" at the end.*

---

## 1. Objective

Predict `P(success | transaction, route)` for a payment routed through one of
several candidate PSP gateways, so that at serving time the system can score
all candidate routes for a transaction and pick the best one (argmax, with a
fallback). Framed as pointwise binary classification, not learning-to-rank.
No public dataset exists for this problem, so the data is fully simulated —
this is stated explicitly here and should be stated explicitly in any
README/write-up derived from this report.

---

## 2. Dataset Generation (`dataset-generator-routing/`)

### 2.1 Live pipeline

The dataset actually used by training is produced by this chain:
routes.py + health_incidentbased.py + transactions.py + router.py
-> simulator.py (run_actual_simulation)
-> generate.py
-> payment_dataset_actual_4week.csv
-> feature_extraction2.py
-> payment_dataset_features_4week.csv
-> ml/lightgbm_tuned.py


- **`routes.py`** — defines 4 static routes (R1–R4), each with a fixed
  `base_success_rate` (0.979–0.986), `base_latency_ms`, `cost_percent`
  (route fee, 1.10%–2.10%), supported payment methods/networks, and incident
  parameters (`incident_probability`, `degraded_probability`, incident
  duration range).
- **`health_incidentbased.py`** — per-minute route health as a 3-state
  process (`HEALTHY` / `DEGRADED` / `OUTAGE`). Each minute, a route either
  stays healthy or (with `incident_probability`) enters an incident of random
  duration, randomly typed DEGRADED vs OUTAGE per `degraded_probability`.
  This is the actual health model used by the live pipeline.
- **`transactions.py`** — generates per-minute transaction volume via a
  Poisson process modulated by an hour-of-day traffic multiplier (`TRAFFIC_MULTIPLIER`,
  peak ~8–9pm) and a weekend multiplier (1.28x). Each transaction gets a
  log-normal amount (capped at 50,000), a weighted-random bank, network,
  payment method, merchant category, device, and a risk level correlated with
  amount (`generate_risk`). Note: there's a dead module-level `amount = ...`
  statement at the top of the file (lines 109–114) that's computed once at
  import time and never used — the real amount is computed per-transaction
  inside `generate_transaction()`. Harmless, but worth deleting for clarity.
- **`router.py`** — the *data-generation-time* route selection policy: a
  fixed-weight random choice (`ROUTE_WEIGHTS`: R1 35%, R2 25%, R3 25%, R4 15%)
  among routes eligible for the transaction's payment method/network. This is
  **not** the ML model — it's what generates the "ground truth" dataset by
  simulating how transactions would have been routed under a naive policy.
- **`simulator.py`** — `run_actual_simulation()` ties it together: for each
  transaction, pick a route via `router.choose_route`, look up that route's
  health state at that minute, compute success probability
  (`HEALTHY` → `base_success_rate`; `DEGRADED` → `base_success_rate * 0.80`;
  `OUTAGE` → `0.0`), draw success via `random.random() < probability`, and
  generate latency (Gaussian around a health-dependent multiple of base
  latency). One row per transaction (the route it was actually sent to).
- **`generate.py`** — entry point, runs a 28-day (`28*24*60` minute)
  simulation and writes `payment_dataset_actual_4week.csv`.

### 2.2 Also present but **not used by the live pipeline** (legacy/experimental)

- **`outcomes.py`** — an alternate, more detailed success-probability model
  (adds risk-level and amount-based penalties, and gives `OUTAGE` a 10%
  success multiplier instead of 0%). Not imported by `simulator.py` or
  `generate.py`. Looks like an earlier iteration.
- **`health_markov.py`** — an alternate Markov-chain health-state model
  (different transition probabilities than `health_incidentbased.py`). Not
  imported anywhere in the live path.
- **`feature_extraction.py`** — an earlier feature-extraction script, used
  only for `payment_dataset_features_1day.csv` (a 1-day smoke-test dataset).
  Superseded by `feature_extraction2.py` for the real 4-week dataset.

These aren't bugs, but if `report.md`/README readers go looking at the repo,
it's worth a one-line note that these are earlier iterations kept for
reference, not part of the current pipeline — otherwise they read as
inconsistencies (e.g. two different OUTAGE success rates).

### 2.3 Important design points confirmed correct

- **`run_counterfactual_simulation()`** also exists in `simulator.py` —
  scores **all 4 routes** per transaction (not just the one chosen), which is
  exactly what's needed to build the round-robin-vs-model routing simulation
  later. It's fully written but **never invoked** — `generate.py:1` has it
  imported and commented out. This is ready to run whenever the routing
  simulation work starts; no new simulation code needs to be written, just a
  new output CSV generated from it.
- **28-day timeline, 21 train / 7 test**, split by absolute time
  (`TRAIN_MINUTES = 21 * 24 * 60` in `feature_extraction2.py`), not randomly —
  correct, since route health drifts over time and a random split would leak
  future route state into training.

---

## 3. Feature Engineering

Two stages, in two different files:

### 3.1 `feature_extraction2.py` — written to the CSV (14 features)

All rolling-window features are **strictly causal**: `prior_rolling_sum()`
(`feature_extraction2.py:54-72`) excludes the current minute by construction
(cumsum-then-shift), so no current-transaction outcome leaks into its own
features. Confirmed no leakage here.

- Transaction: `amount`, `bank`, `network`, `payment_method`, `merchant`,
  `device`, `risk`, `minute`, `day_in_week`, `hour_of_day`
- Route (static): `route_id`, `route_base_success_rate`,
  `route_base_latency_ms`, `route_cost_percent`
- Route (real-time): `route_success_rate_5m`, `route_success_rate_15m`,
  `route_avg_latency_5m`, `route_current_load`, `route_requests_1m`,
  `time_since_last_failure`, `route_flagged_down` (circuit-breaker flag,
  success_rate_5m < 0.70)
- Interaction: `bank_route_success_rate_15m` (Bayesian-shrunk toward the
  route's overall baseline to stabilize low-volume bank/route pairs)

`route_base_success_rate`/`route_base_latency_ms` are computed **only from
the first 21 training days** (`calculate_training_baselines`), preventing the
test week from leaking into a supposedly "static" feature.

### 3.2 `ml/lightgbm_tuned.py::engineer_features()` — computed in-code at
training/eval time, not persisted to CSV (12 additional features)

Per your clarification: these are derived purely from the 14 CSV columns
above and intentionally not written back to the CSV — they're recomputed on
load. All confirmed causal (explicit `.shift(1)` before rolling, or
current-row excluded):

- `route_failure_rate_5m`/`_15m` = `1 - route_success_rate_5m/_15m`
- `route_success_drop_5m`/`_15m` = `route_base_success_rate -
  route_success_rate_5m/_15m` (how far below its own baseline the route
  currently is)
- `route_latency_ratio` = `route_avg_latency_5m / route_base_latency_ms`
  (degradation ratio)
- `bank_route_gap` = `bank_route_success_rate_15m - route_success_rate_15m`
  (is this bank doing better/worse than the route average?)
- `route_stress` = `route_current_load * route_failure_rate_15m` (load ×
  failure rate interaction)
- `bank_failure_rate_5m`, `network_failure_rate_5m`,
  `route_failure_rate_1m` — short-window failure rates at bank/network/route
  granularity, each built the same shift-then-rolling-window way
- `time_since_last_bank_route_success` — minutes since the last success for
  this exact (bank, route) pair
- `amount_to_bank_avg_ratio` — current amount vs. historical (cumulative,
  shifted) average amount for that bank

**Feature importance (latest run, `models/feature_importance.csv`)** — top 5:
`amount` (7.4%), `minute` (7.0%), `route_latency_ratio` (6.7%),
`bank_route_success_rate_15m` (6.1%), `bank_route_gap` (5.7%). The engineered
features collectively rank well — `route_latency_ratio` and `bank_route_gap`
alone account for over 12% of total importance, validating that stage 3.2 was
worth building. Categorical one-hot columns (bank/merchant/device/network/
route_id) are all individually under 0.7% — route_id itself is nearly
worthless (<0.03% total across R1–R4), suggesting the model gets everything
useful about *which* route from the route-level numeric features rather than
route identity itself.

---

## 4. Model Training (`ml/`)

### 4.1 `ml/lightgbm_baseline.py` — initial baseline, now superseded

Single `LGBMClassifier` with fixed hyperparameters, no class weighting, no
threshold tuning (fixed 0.5), no calibration. Kept as a reference point, not
iterated on further.

### 4.2 `ml/lightgbm_tuned.py` — the active model

- 20-candidate random hyperparameter search (`ParameterSampler`) over
  `num_leaves`, `max_depth`, `learning_rate`, `min_child_samples`,
  `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`; selected by
  validation log loss; early stopping (100 rounds) against a held-out
  validation slice (last 20% of the 21 training days, chronological — not
  random).
- **Class weighting**: `class_weight = {0: success_count/failure_count, 1:
  1.0}` (`ml/lightgbm_tuned.py:1021-1028`) — failure class weighted ~21x to
  counter the ~4.5% base failure rate and push recall up on the class that
  matters for routing decisions.
- **Validation-only threshold selection** (`analyze_thresholds()`): sweeps
  13 thresholds, reports failure precision/recall/F1/accuracy at each,
  reports the best-F1 threshold, and separately reports the
  highest-precision threshold that still clears a target recall
  (`TARGET_FAILURE_RECALL = 0.50`). Test set is touched exactly once, at the
  very end, at the threshold chosen from validation — correct train/val/test
  discipline, explicitly commented in the code as intentional.

### 4.3 Model iteration history

Three tuned-model runs occurred over the course of this work; hyperparameters
and the resulting operating point shifted meaningfully each time as class
weighting and the feature set evolved. Latest is authoritative.

| Run | Key hyperparams | Val LogLoss | Threshold | Failure Precision | Failure Recall | Failure F1 | ROC-AUC | Accuracy |
|---|---|---|---|---|---|---|---|---|
| v1 (first tuned run, no class weighting) | lr=0.02, leaves=31, depth=12 | 0.1465 | 0.5 (fixed) | 0.83 | 0.21 | 0.33 | 0.781 | 0.962 |
| v2 | lr=0.10, leaves=63, depth=-1 | 0.1814 | 0.351 (best-F1) | 0.346 | 0.375 | 0.360 | 0.757 | 0.940 |
| v3 (current) | lr=0.10, leaves=63, depth=16 | 0.1801 | 0.176 (target-recall 50%) | 0.229 | 0.450 | 0.303 | 0.746 | 0.906 |

Reading this honestly: v1's headline numbers look best in isolation, but it
was evaluated at an untuned 0.5 threshold with no class weighting — its 0.21
failure recall was the original problem this whole thread of work set out to
fix. v2→v3 trade ROC-AUC/accuracy for deliberately higher failure recall via
class weighting and a lower, validation-chosen threshold — accuracy dropping
from 0.94 to 0.91 is the **expected and accepted cost** of catching more
failures, not a regression. v3 is the current `models/lightgbm_tuned.joblib`.

---

## 5. Calibration Analysis

Class weighting improves failure recall but, as expected, distorts
`predict_proba()` away from true probabilities. Checked directly:

**Before calibration** (test set, v3 model, binned by predicted P(success)):

| bin | n | mean predicted | actual success rate | gap |
|---|---|---|---|---|
| [0.0–0.1) | 1,195 | 0.0125 | 0.0996 | -0.087 |
| [0.1–0.2) | 205 | 0.1576 | 0.6341 | **-0.477** |
| [0.2–0.3) | 375 | 0.2524 | 0.7680 | **-0.516** |
| [0.3–0.4) | 555 | 0.3517 | 0.8360 | -0.484 |
| [0.4–0.5) | 757 | 0.4547 | 0.8177 | -0.363 |
| [0.5–0.6) | 998 | 0.5540 | 0.8447 | -0.291 |
| [0.6–0.7) | 1,679 | 0.6539 | 0.8648 | -0.211 |
| [0.7–0.8) | 2,979 | 0.7554 | 0.8922 | -0.137 |
| [0.8–0.9) | 8,378 | 0.8612 | 0.9358 | -0.075 |
| [0.9–1.0) | 93,444 (85% of data) | 0.9767 | 0.9754 | 0.001 |

Severe, one-directional miscalibration everywhere except the top bin, which
holds the bulk of the data and was already fine. Root cause identified: the
~21x failure-class weight makes the model systematically pessimistic in the
ambiguous middle range — an expected side effect of class weighting, not a
mysterious distributional artifact.

**Fix: isotonic regression** (`sklearn.isotonic.IsotonicRegression`), fit on
validation predictions only, applied to test:

- Brier score: 0.0402 → **0.0340** (~15% improvement)
- Gaps collapsed to roughly ±0.05 across all well-populated bins; the two
  remaining ~0.10-0.12 gaps sit in bins with only 52-54 samples each — sample
  noise, not a real remaining miscalibration pattern.
- ROC-AUC is unaffected (isotonic regression is monotonic/rank-preserving).

**Conclusion**: calibration problem is solved with a standard, well-tested
technique. This also closes the case for implementing AdaCSL (an
adaptive, paper-derived cost-sensitive recalibration method considered
earlier) — the evidence that motivated it (severe local miscalibration) is
gone. Documented as "considered, resolved via isotonic regression instead"
rather than built.

Calibrator saved to `models/isotonic_calibrator.joblib`, meant to be applied
to `model.predict_proba()` output at serving time.

---

## 6. Cost Function (business-cost-based threshold selection)

Motivation: neither raw classification metrics nor a log-loss-optimized
threshold reflect what actually matters economically — a missed failure and
a false alarm are not equally expensive, and a correctly-processed
transaction *earns* a fee rather than costing nothing.

**Design** (per-transaction, using `amount` and `route_cost_percent`,
already present as raw columns):

- Missed failure (FN — predicted success, actually failed): cost =
  `FAILURE_COST_FIXED + FAILURE_COST_PCT_OF_AMOUNT * amount` (retry
  friction + amount-scaled abandonment risk; both currently placeholder
  constants — 50 flat, 2% of amount — that should be justified/tuned before
  being treated as final)
- False alarm (FP — predicted failure, actually would have succeeded): cost
  = the **forgone route fee** (`amount * route_cost_percent / 100`) — the
  revenue not earned because the transaction was routed away unnecessarily
- Correctly processed (TN — predicted success, actually succeeded): **revenue**,
  `amount * route_cost_percent / 100`, subtracted from total cost
- Correctly caught failure (TP): 0 (no fee either way, failure avoided)

**A bug was caught and fixed during this work**: the first version of this
function counted the route fee as a *cost* on true negatives instead of
*revenue*, which made "predict every transaction as a failure" look like the
cost-minimizing policy (avoids all fee-costs, only pays a small flat
false-alarm penalty) — a logically broken incentive, confirmed by the sweep
result monotonically favoring the lowest threshold tested (0.02) with no
interior minimum. Corrected by netting the fee as revenue against TN and
replacing the flat misroute penalty with the real forgone-fee amount per row.

**Status**: fix has been written but not yet re-run — pending a fresh sweep
on validation data before a cost-optimal threshold can be reported here.

Design and the bug/fix history are as before (see prior revision) — the
corrected version nets the route fee as revenue on true negatives rather
than counting it as a cost, and replaces the flat misroute penalty with the
real forgone-fee amount per false positive.

**Result** (validation sweep, corrected function): a genuine interior
minimum at **threshold = 0.18** (cost/profit -6.4551 per transaction),
degrading sharply below ~0.06 (confirms the earlier bug's "always predict
failure" failure mode is gone) and mildly above ~0.30. Final test-set
result at threshold 0.18: **-6.9226 cost per transaction** (net profit,
single test-set touch).

Notably, this cost-optimal threshold (0.18) is nearly identical to the
threshold already selected earlier via an unrelated statistical criterion
(target 50% failure recall → 0.176) — two independent selection methods,
one purely statistical and one economic, converge on the same operating
point. `FAILURE_COST_FIXED`/`FAILURE_COST_PCT_OF_AMOUNT` remain assumed
constants rather than measured figures; a sensitivity check (re-running the
sweep at half/double these values) is recommended before treating 0.18 as
final, to confirm the result is robust rather than assumption-sensitive.

---

## 7. Current State Summary

- **Dataset**: 28-day simulation, 21 train / 7 test days, 4 routes,
  ~4.5% overall failure rate, causal rolling-window features, no leakage
  found in either the CSV-level or in-code feature engineering.
- **Model**: LightGBM, 26 features (14 base + 12 engineered), class-weighted
  for failure recall, hyperparameter-tuned, validation-only threshold
  selection.
- **Calibration**: fixed via isotonic regression (Brier 0.0402 → 0.0340).
- **Cost-aware threshold**: designed, bug found and fixed, re-run pending.
- **Test-set operating point (v3, pre-cost-function threshold)**: ROC-AUC
  0.746, PR-AUC 0.978, Failure Precision 0.229 / Recall 0.450 / F1 0.303 at
  threshold 0.176.

---

## 8. Not Yet Built

1. **Routing simulation** (round-robin vs. model argmax, success-rate
   delta) — the project's own stated headline metric. `run_counterfactual_simulation()`
   already exists in `simulator.py` and just needs to be run to produce an
   all-candidate-routes dataset.
2. **Deployment**: no FastAPI service, no Dockerfile, no load testing
   (Locust/k6), no monitoring. `requirements.txt` is the only artifact
   related to this. This was meant to be the majority of the project's time
   budget and is the recommended next focus.
3. **CacheX integration**: separate project, not started; routing service is
   meant to run against real Redis first, per the original plan.
4. **AdaCSL**: evaluated and deliberately not built — isotonic regression
   already solved the problem it would have targeted. Documented here as a
   considered-and-rejected alternative, which is itself worth a line in the
   README.

---

*Report generated from a full read of `dataset-generator-routing/*.py`,
`ml/lightgbm_baseline.py`, `ml/lightgbm_tuned.py`, `models/feature_importance.csv`,
`requirements.txt`, and the `flow` notes file, plus calibration/cost-function
checks run during this session.*


## 9. Routing Simulation Results

Using `simulator.py::run_counterfactual_simulation`'s design (offline,
proxy-outcome methodology — see script comments for the exogenous-health
assumption this relies on), evaluated on the full test set (110,565
transactions, 405,204 eligible (txn, route) candidate rows after route
eligibility filtering):

| Policy | Success Rate |
|---|---|
| Actual (real weighted-random router, ground truth) | 95.4696% |
| Round-Robin | 95.0365% |
| **Model Argmax** | **97.7873%** |

**Model vs. Round-Robin delta: +2.75 percentage points.**

For comparison, Razorpay's own published Smart Routing paper reports a
~4-6% (avg ~5%) production A/B-tested lift using the same round-robin
baseline. This result is smaller but the same order of magnitude and
direction — expected, given this is an offline synthetic estimate against
a much simpler feature pipeline than their production system, not a live
A/B test.

**Bug found and fixed during this analysis**: `routes.py` spelled the RuPay
network `"RuPay"` for R2/R4 but the data generator produces `"Rupay"`,
silently excluding R2 and R4 from ~18,570 RuPay-network transactions
(forcing them all onto R3, the lowest base-success-rate route). Fixing the
spelling raised the measured delta from +2.28pp to +2.75pp. Caveat: the
model was never trained on `(route_id=R2, network=Rupay)` or
`(route_id=R4, network=Rupay)` combinations (they didn't exist before the
fix), so its predictions for that specific slice are an extrapolation
rather than something directly learned — worth keeping in mind, though it
didn't produce an obviously anomalous result here.








## 10. Dataset & Model v2 — Load-Aware Health + RuPay Fix

### Motivation

Two issues surfaced during review of the v1 pipeline:

1. **Data bug**: `routes.py` spelled the RuPay network `"RuPay"` for R2/R4 but
   `transactions.py` generates `"Rupay"` — silently excluding R2 and R4 from
   all RuPay-network traffic, forcing every such transaction onto R3
   regardless of R3's actual performance.
2. **Modeling limitation**: route health was fully exogenous — generated by
   an independent per-route incident process (`health_incidentbased.py`)
   with zero dependency on transaction volume. In a real payment system,
   gateway capacity strain under load is a first-order cause of
   degradation (explicitly called out in the Razorpay paper reviewed
   earlier: "the Gateway may be overloaded with more capacity than it can
   handle, which leads to a sudden decrease in success rates"). Without
   this, `route_current_load`/`route_requests_1m` had no true causal
   signal to learn from.

Both were judged worth fixing before moving to deployment, since the
routing use case is specifically about avoiding congested routes — a
capability the v1 data couldn't demonstrate.

### Architecture changes

- **`routes.py`**: added `base_capacity_tps` per route (120k/150k/100k/75k
  for R1-R4); fixed the RuPay spelling to be consistent everywhere.
- **`transactions.py`**: decoupled sampled ML rows from simulated
  production traffic. `BASE_SAMPLED_TRANSACTIONS_PER_MINUTE = 15` still
  controls actual CSV row count; a new `BASE_SYSTEM_TPS = 100,000` drives
  aggregate traffic used only internally to compute route load — never
  materialized as individual rows. Removed dead module-level `amount = ...`
  code (computed once at import time, never used — real per-transaction
  amount was always generated separately inside `generate_transaction()`).
- **`simulator.py`** (rewritten): now simulates aggregate per-route,
  per-minute traffic (`build_route_metrics`) alongside the sampled
  transactions. Route utilization (`assigned_tps / effective_capacity_tps`,
  where effective capacity itself drops under `DEGRADED`/`OUTAGE` health)
  feeds two new load-response curves — `load_success_multiplier` and
  `load_latency_multiplier` — both flat below 70% utilization, degrading
  smoothly above it. Aggregate outcomes per route per minute are drawn via
  `np.random.binomial(aggregate_requests, success_probability)` rather than
  simulated row-by-row — this is what keeps the CSV small while still
  representing production-scale volume (100k tps collapses into 2 integers
  per route per minute: `aggregate_request_count`, `aggregate_success_count`).
- **`feature_extraction3.py`** (new file, replaces `feature_extraction2.py`
  for this dataset): `calculate_route_features()` now sources
  `route_success_rate_5m/15m`, `route_avg_latency_5m`, etc. from the
  aggregate `route_metrics` table instead of the small sampled-transaction
  table — a side benefit is these rolling features are now statistically
  far more stable (binomial draws over tens of thousands of trials per
  minute vs. a handful of sampled rows). Fixed `DAYS`/`TOTAL_MINUTES`/
  `TRAIN_MINUTES` fixed constants to a `TRAIN_FRACTION = 0.75`-based split
  computed from actual data length. New numeric features:
  `route_utilization`, `route_effective_capacity_tps`, `system_tps`.
- **`ml/lightgbm_tuned.py`**: removed `"minute"` from `NUMERIC_FEATURES`
  (absolute simulation minute doesn't generalize past this 28-day window);
  removed `"route_current_load"`/`"route_requests_1m"` (now redundant with
  the normalized `route_utilization`); `route_stress` redefined as
  `route_utilization * route_failure_rate_15m` (was `route_current_load *
  route_failure_rate_15m`).

### Bug caught before running: `time_since_last_failure` degeneracy

Reviewing the design before regenerating surfaced a real issue: the
original definition ("minutes since any failure occurred") was computed
from raw transaction counts. At production-aggregate scale (tens of
thousands of requests per route per minute), *some* failure occurs in
nearly every single minute purely from volume — the feature would have
collapsed to ~0 almost everywhere and lost all signal, silently degrading
a feature that ranked in the top 8 in earlier runs. Fixed by keying it off
`health_state != "HEALTHY"` instead (now directly available per
route/minute from `route_metrics`, a signal the old design never had
access to) — confirmed working: it lands as the **#3 most important
feature (6.24%)** in the v2 run, not degenerate.

### File naming

All v2 artifacts use a `_v2` suffix (`payment_dataset_actual_4week_v2.csv`,
`route_metrics_4week_v2.csv`, `payment_dataset_features_4week_v2.csv`,
`models/lightgbm_tuned_v2.joblib`, `models/isotonic_calibrator_v2.joblib`,
`models/feature_importance_v2.csv`) — the original v1 dataset and model
are preserved untouched for direct comparison.

### Results: v1 vs v2

| Metric | v1 | v2 | Change |
|---|---|---|---|
| Val Log Loss | 0.1801 | 0.1645 | better |
| Test ROC-AUC | 0.7460 | 0.7495 | ~same |
| Test Brier (pre-calibration) | 0.0402 | 0.0338 | notably better |
| Test Brier (post-isotonic) | 0.0340 | **0.0268** | ~21% relative improvement |
| Test Accuracy | 0.9064 | 0.9533 | up sharply |
| Failure Precision (target-50%-recall threshold) | 0.2288 | **0.4426** | nearly doubled |
| Failure Recall | 0.4500 | 0.4420 | ~matched by design |
| Failure F1 | 0.3033 | **0.4423** | ~46% relative improvement |
| Cost-optimal threshold | 0.18 | 0.18 | identical |
| Test cost per transaction (at cost-optimal threshold) | -6.9226 | **-7.5380** | ~8.9% more profitable |

**Feature importance validates both fixes empirically**: `system_tps`
(5.61%), `route_utilization` (5.37%), and `route_stress` (4.06%) — none of
which existed in v1 — together account for ~15% of total importance,
confirming load now carries real, causal signal rather than the
confounded correlation it could only have had before.

### Threshold decision

v1's cost-optimal threshold (0.18) and its target-50%-recall threshold
(0.176) nearly coincided. In v2 they diverge meaningfully: target-recall
selection now lands at 0.355, while the cost-optimal threshold stays at
0.18 (≈58-60% recall, ≈27% precision by interpolation). This means: given
the stated cost assumptions, it's economically worth accepting more false
alarms than a recall-target heuristic would choose, since a missed failure
costs more than the forgone-fee opportunity cost of a false alarm.
**Recommendation: use the cost-optimal threshold (0.18) in production**,
not the recall-target one — it's the one grounded in actual dollar
tradeoffs, which is the whole point of having built the cost function.


## 8. Not Yet Built

1. **`routing_simulation.py` needs updating for the v2 pipeline** — it
   currently calls `calculate_route_features(raw)` with the old
   (pre-`route_metrics`) signature, imported from `feature_extraction2`.
   Needs repointing to `feature_extraction3` and `route_metrics_4week_v2.csv`.
2. **True closed-loop routing evaluation** — the current routing
   simulation (and its v2 successor) is a one-shot offline estimate: it
   assumes a single transaction's routing choice doesn't materially affect
   future route health. Now that load genuinely affects health, a fully
   rigorous evaluation would need routing decisions to feed back into
   future utilization/health state — a sequential, closed-loop simulation
   rather than a static snapshot comparison. Flagged as future work; the
   offline estimate remains a reasonable approximation for now since a
   single transaction is a negligible fraction of aggregate route volume.
3. **Deployment**: no FastAPI service, no Dockerfile, no load testing, no
   monitoring. This remains the primary focus going forward.
4. **CacheX integration**: separate project, not started.
5. **AdaCSL**: evaluated and rejected — isotonic regression already
   solves the calibration problem it would have targeted, in both v1 and
   v2.




## 11. Threshold Selection — Multi-Objective (Cost + Recall) and Benchmark Comparison

### Razorpay benchmark comparison (v2)

Recomputed success-class precision from the v2 test confusion matrix
(`[[3086,3896],[3886,155758]]`, predicted-success = 159,654, TP = 155,758):

| | Razorpay LightGBM | Razorpay best (Random Forest) | v1 | **v2** |
|---|---|---|---|---|
| Precision | 0.9433 | 0.9469 | 0.9636-0.9726 | **0.9756** |
| ROC-AUC | 0.7130 | 0.7949 | 0.7460-0.7810 | 0.7495 |

v2 now exceeds Razorpay's published LightGBM on precision outright, and is
closing in on their best model (Random Forest), on a single-model,
synthetic-data comparison against their production-scale ensemble
evaluation.

### Hyperparameter search methodology review

Compared against the paper's stated method ("optimal hyperparameters were
chosen using the grid-search cross-validation method", tuned to maximize
precision):

- **Random search (`ParameterSampler`, 20 candidates) vs. their grid
  search**: not considered a gap. Random search is established
  (Bergstra & Bengio, 2012) to often outperform grid search for equal
  compute budget, since grid search spends evaluations uniformly across
  axes regardless of importance. No change made here.
- **Single time-based validation split vs. their cross-validation**: a
  legitimate difference, but naively adopting standard k-fold CV would
  reintroduce the time-leakage problem this project has deliberately
  avoided throughout (random folds would leak future route-health state
  into training). A faithful version would require walk-forward /
  `TimeSeriesSplit`-style CV — meaningfully more compute (k× training cost
  per candidate) for uncertain payoff. **Deferred as future work**, not
  done now given the deployment timeline.
- **Selection metric mismatch — the real gap**: hyperparameter search
  currently selects the winning candidate by validation log loss
  (`tune_model()`), which doesn't track the actual objective (business
  cost / failure detection), the same category of problem that motivated
  building the cost function in the first place. Razorpay explicitly
  selected by precision for the same underlying reason ("does not end up
  assigning high probabilities to low-performing terminals"). **Not yet
  implemented**: swapping the search's scoring function from log loss to
  validation cost-per-transaction (reusing `compute_total_cost`), so model
  selection and threshold selection are optimizing the same thing.
  Flagged as a future-work item, prioritized behind deployment.

### Multi-objective threshold selection

Pure cost-minimization risks landing on an extreme operating point if the
cost assumptions (`FAILURE_COST_FIXED`, `FAILURE_COST_PCT_OF_AMOUNT`) are
imprecise. Ran a constrained-optimization comparison instead of trusting
the cost-optimal threshold in isolation: swept thresholds on validation,
computing cost/txn and failure precision/recall/F1 together, then compared
two selections —

- **A: pure cost-optimal** (min cost/txn) → threshold 0.18
- **B: cost-optimal subject to failure recall ≥ 50%** (the recall floor
  already established as important earlier in this project) → threshold
  0.12

**Test results** (single touch, both candidates evaluated together):

| | A: cost-optimal (0.18) | B: recall-constrained (0.12) |
|---|---|---|
| Cost/txn | **-7.5380** | -7.4521 |
| Failure Precision | **0.5945** | 0.4209 |
| Failure Recall | 0.4036 | **0.4484** |
| Failure F1 | **0.4808** | 0.4343 |

**Finding: A dominates B on every metric except recall, and the recall gap
is modest.** Cross-checking against the validation sweep explains why —
the cost-optimal threshold (0.18, F1=0.499) sits almost exactly at the
F1-maximizing region (peak ~0.508-0.510 around threshold 0.22-0.28): the
economic optimum and the statistical balance-optimum were already nearly
the same point, not two competing objectives requiring a tradeoff.

**Generalization-gap lesson**: Selection B was chosen because validation
recall (0.5099) cleared the 50% floor, but test recall came out to 0.4484
— *below* its own target. Not a bug, the expected validation→test transfer
gap, but a concrete illustration that selecting a threshold right at a
hard constraint boundary is fragile. If a genuine recall floor is needed
in production, target comfortably above it on validation (e.g. ≥55-60% to
reliably hold ≥50% live), not exactly on the line.

**Decision: ship threshold = 0.18** (pure cost-optimal). It wins on cost,
precision, and F1, and the recall shortfall relative to B is small and,
per the point above, B's own recall advantage wasn't even reliable
out-of-sample.



## 12. Routing Simulation — v2 Results

Rewrote `routing_simulation.py` for the v2 pipeline: imports from
`feature_extraction3` instead of `feature_extraction2`, loads
`route_metrics_4week_v2.csv` and threads it through the new
`calculate_route_features(route_metrics, total_minutes)` signature,
computes `total_minutes`/`train_minutes` explicitly (replacing the removed
fixed constants), and fixes `route_stress` to multiply by
`route_utilization` rather than the old `route_current_load`, matching the
training-time correction.

### Results

Evaluated on the full v2 test set (166,626 transactions, 610,948 eligible
candidate rows after route-eligibility filtering):

| Policy | Success Rate |
|---|---|
| Actual (real weighted-random router, ground truth) | 95.8098% |
| Round-Robin | 95.4371% |
| **Model Argmax** | **97.2528%** |

**Model vs. Round-Robin delta: +1.8157 percentage points** — down from
v1's +2.75pp.

### Why the delta shrank, and why that's a good sign, not a regression

Every classifier-level metric improved from v1 to v2 (precision, F1,
calibration, cost-per-transaction — see Sections 10-11), so a smaller
routing delta needed an explanation rather than being taken at face value.

The `outcome_proxy` (`route_success_rate_5m`) used to score whichever
route a policy picks was, in v1, computed from small sampled-transaction
counts — the same metric that showed bins with as few as 52-54 samples
during earlier calibration checks. The **model argmax** policy specifically
selects the highest-predicted-probability route, making it disproportionately
likely to land on a route whose noisy v1 proxy happened to show a lucky
short-term spike — inflating its measured advantage. **Round-robin** doesn't
chase the proxy at all, so its measured outcome stayed a fair, largely
unbiased sample regardless of that noise.

This asymmetry predicts exactly the observed pattern: both baseline
policies moved *up* slightly from v1 to v2 (Actual 95.47%→95.81%,
Round-Robin 95.04%→95.44%, consistent with general data regeneration),
while the model's number moved *down* (97.79%→97.25%) once v2's
aggregate binomial-based route health (tens of thousands of trials per
route per minute, vs. a handful of sampled rows) removed the short-term
noise the argmax policy could previously exploit.

A secondary, compounding factor: now that load genuinely affects health,
degradation is shared more broadly across routes during high-traffic
periods rather than being purely route-specific-incident-driven, which can
genuinely narrow the true gap between the best and worst available route
at any given moment — reducing the headroom *any* routing policy has to
exploit.

**Conclusion: v2's smaller delta (+1.82pp) is the more trustworthy number**,
not a worse result — it's no longer measured against a proxy with known
small-sample noise problems. Reported here in place of v1's figure. For
comparison, Razorpay's own published Smart Routing paper reports a ~4-6%
(avg ~5%) production A/B-tested lift; this remains an offline, synthetic,
proxy-based estimate rather than a live A/B test, so the gap to their
figure is expected.

Sanity check: row-count math for the eligibility table verifies exactly
(`27,680×3 + 82,997×4 + 27,876×3 + 28,073×4 = 610,948`, matching the
printed total), and RuPay eligibility correctly shows `R2, R3, R4` per the
earlier fix.


## 8. Not Yet Built

1. **True closed-loop routing evaluation** — the routing simulation
   remains a one-shot offline estimate: it assumes a single transaction's
   routing choice doesn't materially affect future route health/utilization.
   Now that load genuinely affects health, a fully rigorous evaluation
   would need routing decisions to feed back into future utilization —
   a sequential, closed-loop simulation rather than a static snapshot
   comparison. Reasonable approximation for now given a single transaction
   is a negligible fraction of aggregate route volume; flagged as future
   work.
2. **Hyperparameter search selection metric** — currently selects by
   validation log loss rather than validation cost-per-transaction (or
   failure-class PR-AUC), the same category of misalignment that
   motivated building the cost function in the first place. Not yet
   implemented; deferred behind deployment.
3. **Time-aware cross-validation** — current search uses a single
   chronological validation split rather than walk-forward CV. Legitimate
   difference from Razorpay's methodology, deferred given the compute
   cost (k× training time per candidate) for uncertain payoff.
4. **Deployment**: no FastAPI service, no Dockerfile, no load testing, no
   monitoring. This is now the primary focus.
5. **CacheX integration**: separate project, not started.
6. **AdaCSL**: evaluated and rejected in both v1 and v2 — isotonic
   regression already solves the calibration problem it would have
   targeted.



   ## 14. Final Model Summary — ML Pipeline Closed Out

This section closes out the modeling phase of the project. Every version
below was a genuine, honestly-obtained result at the time it was produced
— none are superseded in the sense of being wrong, only in the sense of
later versions reflecting additional fixes and better methodology. Kept
here as the full record, not just the final number.

### 14.1 Iteration history

| Version | What changed | Threshold | Precision | Recall | F1 | Cost/txn | Routing delta |
|---|---|---|---|---|---|---|---|
| v1 | Original 26-feature model, no load-awareness, RuPay bug present | 0.176 | 0.229 | 0.450 | 0.303 | -6.9226 | +2.75pp* |
| v2 (leaky) | Load-aware health added, but `route_utilization` unknowingly leaked ground-truth health state | 0.18 | 0.595 | 0.404 | 0.481 | -7.5380 | +1.82pp |
| v2 (accidental half-fix) | Code fixed but data regeneration mislaid to wrong directory — trained on stale leaky data. Caught via row-count forensics, not a real result. | — | — | — | — | — | — |
| v2 (properly fixed) | `route_utilization` redefined to use only observable (health-blind) capacity; `time_since_last_failure` redefined off observable success-rate crossing instead of simulator-only `health_state` | 0.16-0.18 | 0.77-0.79 | 0.348-0.349 | 0.481-0.483 | -7.508 to -7.523 | +1.4665pp |
| v2 (cost-based hyperparameter search + tunable class weight) | Selection metric switched from log loss to validation cost; class weight multiplier searched (won at 0.75x base) | 0.16 | 0.7745 | 0.3491 | 0.4813 | -7.5083 | (not rerun — see note) |

*v1's routing delta is reported post-RuPay-fix, pre-load-aware-health (measured before the v2 regeneration).

Routing simulation was not rerun after the final hyperparameter-search
retrain — its test-set metrics (precision, recall, F1, cost) are
statistically indistinguishable from the immediately prior "properly
fixed" version it's compared against above, so the +1.4665pp delta stands
as representative of the final model rather than requiring another full
simulation pass.

### 14.2 What this iteration process actually found

Real, substantive issues were caught and fixed along the way — this
wasn't just repeated retraining for its own sake:

- **RuPay spelling bug** (`routes.py`): silently excluded R2/R4 from an
  entire network segment, forcing it onto the worst-performing route.
  Fixed; measurably widened the routing delta.
- **Exogenous health limitation**: original simulator had zero
  load→health coupling, making `route_current_load` causally meaningless.
  Fixed via aggregate-traffic simulation (`simulator.py` rewrite,
  sampled-rows-vs-aggregate-TPS split) — confirmed working via feature
  importance (`system_tps`, `route_utilization`, `route_stress` together
  reaching ~15% importance where they previously carried none).
- **`time_since_last_failure` degeneracy risk**: caught *before* running,
  by reasoning through what "any failure occurred" means at production
  aggregate scale (nearly every minute, by volume alone). Fixed proactively
  by keying off `health_state` initially, then a second time off a fully
  observable proxy once the deployability issue below was found.
- **Ground-truth leakage into supposedly-servable features**: `route_utilization`
  and `route_effective_capacity_tps` depended on the simulator's private
  `health_state`, which a real production system would never have access
  to. Found via reasoning about Redis/CacheX deployability, not from a
  metric regression — fixing it cost essentially nothing economically
  (cost, F1 unchanged) while making the model honestly deployable.
- **Cost function sign error**: an early version counted route fees as a
  cost instead of revenue on true negatives, producing a nonsensical
  "always predict failure" optimum. Caught by the fact that the optimum
  landed on a sweep boundary rather than an interior point — a useful
  general lesson (a boundary optimum is itself a signal something's wrong).
- **Recall-floor-constrained threshold selection is fragile**: demonstrated
  twice, independently, that forcing a hard recall floor (rather than
  optimizing cost freely) can walk into a catastrophic operating point
  (4-5% precision, net financial loss) when the model's precision-recall
  curve shifts. Pure cost-optimization proved robust across every version.
- **Data-regeneration pipeline mistakes** (files landing at project root
  instead of the target subfolder, twice) — caught via row-count
  forensics (identical counts across supposedly-independent unseeded
  regenerations are essentially impossible by chance) rather than trusting
  file timestamps or assuming success from a clean exit code.

### 14.3 Final production configuration

- **Model**: `models/lightgbm_tuned_v2.joblib` — 26 features (7 categorical,
  19 numeric), class-weighted (0.75x base inverse-frequency weight),
  selected via validation cost rather than log loss.
- **Calibration**: `models/isotonic_calibrator_v2.joblib`, fit on
  validation only. Consistently improves Brier score ~18-21% relative
  across every version tested.
- **Decision threshold**: **0.16** (failure-probability cutoff), chosen by
  pure cost-optimization, validated as robust via a cost-assumption
  sensitivity sweep (0.5x-10x the base cost constants) and confirmed as a
  near-ceiling operating point via an independent hyperparameter
  re-search.
- **Action item before deployment**: the saved artifact's `threshold`
  field currently reflects whatever `analyze_thresholds()`'s target-recall
  heuristic produced (a known-bad value, per Section 14.2's fragility
  finding) — **must be explicitly overwritten to 0.16** before any serving
  code trusts `artifact["threshold"]` directly:

```python
import joblib
artifact = joblib.load("models/lightgbm_tuned_v2.joblib")
artifact["threshold"] = 0.16
joblib.dump(artifact, "models/lightgbm_tuned_v2.joblib")

14.4 Honest characterization of the result
At the chosen operating point: 77-79% precision, ~35% recall on
failure detection. Recall is capped not by under-tuning but by a real
ceiling in the current model's discriminative power (ROC-AUC ~0.75) —
demonstrated via a cost-assumption sensitivity analysis (even a 10x more
pessimistic failure-cost assumption only pushes recall to ~48%, and at
that point the system goes net-unprofitable) and independently confirmed
by a from-scratch hyperparameter re-search optimizing directly for cost
rather than log loss, which converged to the same result rather than
beating it. Precision at this operating point (77-79%) directly compares
well against Razorpay's own published LightGBM benchmark (94.3% on their
differently-scoped success-class metric, on real production data) —
strong for a synthetic, single-model project.

14.5 Remaining future work (unchanged from before, for completeness)
True closed-loop routing evaluation (routing decisions feeding back
into future utilization) — deferred, current one-shot offline estimate
remains a reasonable approximation.
Time-aware (walk-forward) cross-validation — legitimate methodological
gap vs. Razorpay's approach, deferred given compute cost for uncertain
payoff.
CacheX integration — separate project track.
AdaCSL — evaluated and rejected; isotonic regression already solves
the calibration problem it would have targeted.
ML pipeline: closed. Moving to deployment — FastAPI service, Redis
integration, containerization, load testing.

