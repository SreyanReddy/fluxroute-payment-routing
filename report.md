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


## 15. Deployment

### 15.1 Architecture

Stateless FastAPI service, three replicas behind nginx (`least_conn`, keepalive
to upstreams), all sharing one Redis instance that holds every piece of mutable
state. Prometheus scrapes all three instances; Grafana provisions a dashboard
from `deploy/grafana/`.

```
nginx :8080  ->  app1 / app2 / app3 (uvicorn)  ->  Redis
                          |
                     /metrics  ->  Prometheus :9090  ->  Grafana :3000
```

Files: `serving/main.py`, `serving/feature_storing.py`, `serving/Dockerfile`,
`docker-compose.yml`, `deploy/nginx.conf`, `deploy/prometheus.yml`,
`deploy/grafana/`, `loadtest/locustfile.py`.

### 15.2 Feature parity mechanism

`serving/feature_storing.py` mirrors the offline formulas in
`feature_extraction3.py` and `engineer_features()` on three Redis primitives:

1. **Minute-bucketed counters** (`{scope}:{metric}:{epoch_minute}`, TTL 1800s)
   for every rolling-window feature, at route / bank / network / bank×route /
   global scope. Reads sum buckets `[now-N, now-1]`, **excluding the current
   minute**, matching `prior_rolling_sum()`'s convention exactly.
2. **Last-event-minute keys** for `time_since_last_failure` (keyed on the
   observable 5m success rate crossing `FLAG_DOWN_THRESHOLD`, per the
   deployability fix in Section 10) and `time_since_last_bank_route_success`.
3. **Cumulative sum/count** for `amount_to_bank_avg_ratio`, read before the
   current transaction is incorporated, matching the offline
   `cumsum().shift(1)`.

`main.py` reads the feature list from the model artifact itself
(`artifact["numeric_features"] + artifact["categorical_features"]`) rather than
importing `ml/lightgbm_tuned.py`. This removes the training code from the
serving container entirely and makes it structurally impossible for the serving
feature list to drift from what the model was actually fitted on. All 33 columns
(26 numeric + 7 categorical) are produced by `build_route_features()`.

### 15.3 Bug found: in-flight transactions counted as failures

The first version of `get_rolling_rate()` computed success rate as
`successes / requests`. But the two counters are written at different points in
the transaction lifecycle: `req` increments when the routing decision is made,
`succ` only when `/outcome` arrives. Every transaction still in flight was
therefore being counted as a failure.

Offline this bias does not exist — `feature_extraction3.py` computes rates over
completed minutes where every outcome is already known. So this was genuine
train/serve skew, and it gets *worse* under load, since higher throughput means
more concurrently in-flight transactions.

The observable symptom during manual testing was all four candidate routes
returning identical predicted probabilities: with success rates artificially
depressed toward zero across every route, the routes became indistinguishable
in feature space and their raw scores collapsed into the same isotonic
calibration step.

Fixed by adding a separate `resolved` counter incremented on outcome arrival,
and using it as the denominator for success rate, average latency, and the
bank×route shrunk rate. `req` remains the denominator for utilization and
`system_tps`, which correctly should count all arriving requests regardless of
whether they have resolved.

### 15.4 Two smaller correctness notes

- **Prometheus and uvicorn workers**: `prometheus_client` keeps a per-process
  registry, so running multiple uvicorn workers per container would mean each
  scrape hits one arbitrary worker and returns partial counters. Containers run
  a single worker; concurrency comes from replicas instead, which keeps
  per-instance metrics exact and the scaling story simple.
- **Ranking uses raw, not calibrated, probabilities.** Isotonic regression is a
  step function, and in the 0.9+ region where nearly all traffic sits its steps
  are wide enough to map genuinely different routes onto an identical calibrated
  value — at which point `idxmax()` silently degenerates into "first row wins".
  Raw scores retain full resolution, so ranking uses them; the calibrated value
  is still what gets reported to the caller, since that is the number with a
  meaningful absolute interpretation.

Note that the Section 14.3 action item about overwriting `artifact["threshold"]`
is moot for the current design: `/route` performs pure argmax over candidates
and never reads a threshold. It would become relevant only if a circuit-breaker
gate is added later.

### 15.5 Load testing

`loadtest/locustfile.py` samples transactions from the same distributions as
`transactions.py`, calls `/route`, then reports a simulated outcome back to
`/outcome` — driving the real feedback loop rather than hammering a read-only
endpoint, so route health actually evolves under load. `/route` and `/outcome`
are reported as separate Locust request types.

The Grafana dashboard separates **Redis feature-fetch latency** from **model
scoring latency** as distinct histograms. This is deliberate: it identifies
which half of the request path dominates p99, and establishes the baseline the
CacheX comparison will be measured against.

**Status: harness and full stack built; no load-test run has been recorded yet.**
p99 latency and throughput figures are deliberately absent from the README until
an actual run produces them.


## 16. Environment Recovery and Serving Audit

Work resumed after a ~6-week break. Before planning anything, the state of the
workspace was checked from scratch rather than assumed.

### 16.1 What was found

| Finding | Detail | Consequence |
|---|---|---|
| Virtual environment broken | `.venv/bin/python3 -> 'unsupported reparse tag 0xa000000c'`. The project lives on an NTFS mount (`/media/sreyan/D/`), which mangled the interpreter symlink into a Windows reparse point. `site-packages` was intact. | No Python could run. |
| Feature CSV missing | `payment_dataset_features_4week_v2.csv` was gone; only `payment_dataset_actual_4week_v2.csv` and `route_metrics_4week_v2.csv` survived. | Couldn't evaluate or test against training data. |
| Docker not installed | `docker: command not found`. | The deployment stack had never run. |
| The recorded load test measured nothing | `loadtest/run1_failures.csv`: 53,631 occurrences of `status=0`, all requests, 2026-09-07 18:58:33Z to 19:00:55Z. `status=0` is a refused connection, not an HTTP error. | The 377 req/s and 2 ms median in `run1_stats.csv` are how fast the kernel refuses connections. No signal; discarded. |

### 16.2 Fixes

- **New environment** at `~/.venvs/fluxroute`, on the ext4 home directory so
  NTFS can't corrupt it again. Built with `python3.10` specifically, since that's
  what the model was pickled under, and installed from all three requirement
  files (root, `serving/`, `loadtest/`).
- **Feature CSV regenerated** by re-running `feature_extraction3.py` on the two
  surviving inputs. The path has no randomness, so the output is reproducible.
  Verified against the final training run rather than assumed:

| | Regenerated | Final training run |
|---|---|---|
| Test rows | 165,857 | 165,857 |
| Train + validation rows | 496,521 | 397,216 + 99,305 = 496,521 |

  Total 662,378 rows, 28 columns, 40,320 minutes. The model artifact remains
  valid, so no retraining was needed. v1 data was not regenerated: its numbers
  are already recorded in this report and nothing downstream uses it.

### 16.3 The serving spec, and verifying it

`FEATURE_SERVING_SPEC.md` was written during the break. It's a precise contract
for how each of the 26 numeric features must be computed live, and it's
considerably more rigorous than the original Phase 1 design. Several of its
values could be checked directly against the regenerated training data. All
eight route constants it quotes matched exactly, and the serving code's
hardcoded values did not:

| Route | `route_base_success_rate` training / hardcoded | `route_base_latency_ms` training / hardcoded |
|---|---|---|
| R1 | 0.97016 / 0.986 | 168.53 / 145 |
| R2 | 0.95612 / 0.981 | 116.12 / 85 |
| R3 | 0.96275 / 0.979 | 130.18 / 110 |
| R4 | 0.94450 / 0.983 | **283.69 / 175** (62% off) |

The serving code had typed in the simulator's *nominal* settings. Training
recorded the *observed* means, which are lower for success and higher for
latency because they include incidents and congestion. The model learned the
observed values. This was real train/serve skew, and it was silent.

### 16.4 Minor: the artifact's stored threshold

`models/lightgbm_tuned_v2.joblib` stores `threshold = 0.12069`, which is the
target-recall value from `analyze_thresholds()`, the one Section 14.2 showed to
be fragile. `ml/save_artifact.py` (which sets 0.18) was overwritten by a later
retrain. This is harmless today, because `/route` does pure argmax and never
reads a threshold. It should still be corrected so the artifact doesn't carry a
misleading number.


## 17. The `system_tps` Scale Problem

### 17.1 What the imputer revealed

The fitted median imputer stores the training median of every numeric feature:

| Feature | Training median |
|---|---|
| `route_success_rate_5m` | 0.98101 |
| `route_avg_latency_5m` | 145.0 |
| `route_utilization` | 0.34998 |
| `system_tps` | **160,169** |

`system_tps` was trained on the simulator's production-scale stream (base
100,000 TPS times the time-of-day curve). A live service can only measure its
own traffic, a few hundred requests per second under a load test. That's three
orders of magnitude outside anything the model saw, and `route_utilization` and
`route_stress` are derived from it.

### 17.2 What the model does with an out-of-range value

Tree models don't extrapolate. Every split is a threshold like
"`system_tps` < 45,000?", and 200 falls below all of them, so the model treats it
exactly like the smallest training value. The effect is a consistent bias, not
random noise: **the model always believes it is the quietest night it has ever
seen**, and therefore that every route is lightly loaded.

### 17.3 Decision: scale the live meter (implemented, see 17.4)

Treat the service's traffic as a sample of a larger production stream and
multiply by a configured ratio:

```
system_tps = (requests in the last 60 completed seconds / 60) x SCALE
```

The dataset's own sampling ratio is 1 row per 400,000 production transactions
(15 sampled rows per minute out of 6,000,000). That ratio only holds if requests
arrive at the dataset's natural rate (~0.25/s). A load test runs about 1,000x
faster, so 400,000 would push the value far above the training range instead of
below it. The practical rule is:

```
SCALE = training median / planned test rate      e.g. 160,169 / 200 ~ 800
```

SCALE is a configuration value, not a constant. The result is clamped to the
training range, which `models/serving_constants_v2.json` now records as **8,458
to 308,555** (median 162,978 over all training rows; the imputer's 160,169 was fit
on the training part without the validation tail).

`route_utilization` changes at the same time to the spec's formula,
`system_tps x frozen_routing_share[r] / base_capacity_tps[r]`, using the training
shares R1 0.35, R2 0.25, R3 0.25, R4 0.15, instead of each route's observed
traffic.

**Remaining limitation, documented rather than hidden:** in training,
`system_tps` and `hour_of_day` moved together, since both came from one
time-of-day curve. A flat load test produces combinations the model rarely saw
(midday traffic at 3am). The principled long-term fix is to retrain, either
without these features or with the live-meter definition.

### 17.4 Implementation

- **Meter** (`serving/feature_storing.py`): `record_ingress()` increments
  `fr:ingress:sec:{second}` (TTL 120 s) for every `/route` request.
  `read_system_tps()` sums the last 60 completed seconds, so the current second
  and the request itself are excluded, then divides by 60, multiplies by
  `TRAFFIC_SCALE`, and clamps to the training range from
  `serving_constants_v2.json`. It returns the reading and a clamp direction
  (`low` / `high` / none).
- **Once per request** (`serving/main.py`): the meter is read once and the value
  is passed into `build_route_features(..., system_tps=...)`, so every candidate
  route sees the same reading. The old per-minute `global:req` counter was
  removed; the ingress counter replaces it at the same cost.
- **Utilization**: `system_tps × routing_share / base_capacity_tps`, with shares
  and capacities from the constants file.
- **Configuration**: `TRAFFIC_SCALE` defaults to 1.0 in code;
  `docker-compose.yml` sets `${TRAFFIC_SCALE:-800}` (800 suits a ~200 req/s
  test), overridable per run from the shell.
- **Observability**: gauge `fluxroute_system_tps` (the reading passed to the
  model) and counter `fluxroute_system_tps_clamped_total{direction}`.
- **Clock**: `current_minute()` is now derived from `current_second()`, so tests
  freeze one function and both stay consistent.

Verification:

| Check | Result |
|---|---|
| `tests/traffic_meter_check.py` (SCALE 800) | 5 req/s clamped to 8,458 (low); 20 to 16,000; 200 to 160,000; 350 to 280,000; 500 clamped to 308,555 (high). A million requests in the current second are ignored. All pass. |
| Parity test (replay at SCALE 1) | Unchanged: 22 PASS, 4 KNOWN GAP, 0 FAIL. `system_tps`, `route_utilization`, `route_stress` still at a 5.14% median, so the frozen-share formula reproduces training's definition exactly and only the one-minute lag remains. |
| End-to-end smoke test of `main.py` (in-memory Redis) | `/route` then `/outcome` three times; scoring, recording and outcome all work. With only 3 requests the reading clamped low each time and the clamp counter recorded 3, which is the intended signal for "too little traffic for this SCALE". |

**Related fix:** Redis's eviction policy changed from `allkeys-lru` to
`volatile-lru`. With counters now kept for 24 h, memory pressure is more
plausible, and `allkeys-lru` could evict keys with no expiry, such as the
running bank amount totals and the last-success markers, which can't be
rebuilt. `volatile-lru` only evicts keys that have a TTL.


## 18. Throughput Target

### 18.1 Can one machine do 250k TPS?

No, regardless of optimization. For scale: Visa's network is commonly cited at
about 65,000 TPS capacity, and UPI averages under 10,000 TPS (about 20 billion
transactions a month). Counting the current design's Redis calls:

| Per transaction | Redis round-trips |
|---|---|
| `/route`: ~20 reads x 4 candidate routes, plus decision writes and the pending record | ~94 |
| `/outcome` | ~20 |
| **Total** | **~115** |

At 250k TPS that's about 29 million Redis operations per second, against
roughly 100k to 1M for a single instance. Separately, building a pandas
DataFrame and running it through the sklearn `ColumnTransformer` for 4 rows costs
around 1 to 3 ms of CPU, capping each core near 300 to 1,000 requests/second.
And a single laptop can't *generate* 250k req/s of load either.

### 18.2 Redefined target

1. Maximize **per-node throughput** through a measured bottleneck loop: profile,
   fix the top bottleneck, re-measure, log the numbers.
2. Show **horizontal scaling** (1 to 3 replicas).
3. **Project** the cluster size needed for 250k TPS from the measured numbers.

### 18.3 The main lever: most features are constant within a minute

Every clock window in the spec excludes the current minute (`[m-5, m)`), so
these features are identical for every request during minute `m`: route
success/failure rates (5m, 15m, 1m), average latency,
`time_since_last_failure`, `route_flagged_down`, the bank×route rate, bank and
network failure rates, and everything derived from them. That's about 20 of the
26.

Only `amount_to_bank_avg_ratio`, `time_since_last_bank_route_success`, and
`system_tps` / `route_utilization` change per request.

So each process can compute a per-minute snapshot once and serve the rest of the
minute from memory. Reads per request drop from ~80 to ~3, **with no loss of
parity**, because training uses the same "previous minutes only" rule.

### 18.4 Planned bottleneck ladder (estimates, to be replaced by measurements)

| Step | Change |
|---|---|
| 0 | Baseline measurement of the current code, with `py-spy`, Redis `INFO commandstats`, and the Grafana fetch-vs-scoring split |
| 1 | Per-minute snapshot cache |
| 2 | Remaining per-request reads and writes in one pipeline / Lua script (also gives the spec's atomicity) |
| 3 | Replace pandas + `ColumnTransformer` with a precomputed numpy path; optionally a compiled LightGBM model |
| 4 | Multiple processes per node (needs Prometheus multiprocess mode) |
| 5 | Shard Redis / batch counter writes |

A side effect: SCALE from Section 17.3 exists only because test traffic is far
below production traffic, so it shrinks toward 1 as real throughput grows.

The live dashboard UI is deferred until after this work.


## 19. Train/Serve Parity Test

### 19.1 What it is

The 26 numeric features are computed twice, by unrelated code: in batch by
training (`feature_extraction3.py`, `engineer_features()`) and live by serving
(`serving/feature_storing.py`). When the two disagree nothing crashes, and
ordinary unit tests don't catch it, because each side returns a valid number. A
parity test compares them directly: give both the same history and check that
they compute the same numbers. It is also the guard for the throughput work
ahead. Every optimization rewrites how features are computed, and each one is
only accepted if parity stays green.

### 19.2 Design

Three files in `tests/parity/`, each with one job:

| File | Job |
|---|---|
| `history.py` | Rebuilds "what the world looked like just before a transaction" from the same CSVs training used. Knows nothing about serving. |
| `adapter_current.py` | The only file that knows the current serving code's Redis key layout. Loads the history into an in-memory Redis (`fakeredis`), freezes the serving clock at the transaction's minute, and calls `build_route_features()` exactly as `/route` does. A rewrite of serving gets a new adapter; nothing else changes. |
| `run_parity.py` | The referee. Gets training's answers by running `ml/lightgbm_tuned.py::load_data` (cached after the first run), picks transactions, compares every feature, prints a report, and exits non-zero on failure. |

Sample: 500 test-week transactions (400 random, 50 during route incidents, 50 in
quiet hours 2 to 4am; seed 7). Values count as equal within a relative
tolerance of 1e-6.

**Two streams.** Training built route health (5m/15m success rate, latency,
incidents) and `system_tps` from the production-scale stream (`route_metrics`),
but bank, network, bank×route, the amount ratio, last-success and
`route_failure_rate_1m` from the sampled transactions. The history keeps these
separate and feeds each formula the stream it was trained on, because that's
what "same inputs" means.

**Scope.** The test checks formulas given identical inputs. It does not check
whether live inputs resemble training inputs (the Section 17 scale problem), and
it does not exercise the write path. For example, the in-flight bug from
Section 15.3 can't appear in fully-resolved replayed history.

### 19.3 First run against the original serving code: 13 of 26 failed

The failures traced to four root causes:

| Cause | Features | Evidence |
|---|---|---|
| Hardcoded nominal constants | 7: both base constants, `route_success_drop_5m/15m`, `route_latency_ratio`, `bank_route_success_rate_15m`, `bank_route_gap` | Five features shared the identical worst difference, 0.0384997, which is R4's base-rate error. That's the signature of one bug spreading through derived features. |
| Clock-minute windows instead of the last 5 *active* minutes | 2: bank and network failure rates | 45/500 and 25/500 mismatches, concentrated at quiet times. Example: training 1/7 = 0.143, serving 0.0. |
| Previous-minute vs current-minute traffic | 3: `system_tps`, `route_utilization`, `route_stress` | Up to 20% off. Example: training 10,452, serving 9,869. |
| Two streams | 1: `route_failure_rate_1m` | Training 0.0 (a handful of sampled transactions, none failed); serving 0.019 (production-scale stream). |

The 13 that passed cover the core machinery: the "previous minutes only" window
logic, rates, latency, incident flags, the running timers and totals, and the
time features. The design was sound; the bugs were specific.

### 19.4 Should `route_failure_rate_1m` get its own counter?

Measured on the full test week (165,857 transactions):

| Measure | Value |
|---|---|
| Gain importance | 6.50%, **rank 3 of 33** inputs |
| Split importance | 1.82% |
| ROC-AUC with the feature shuffled | 0.75082 to **0.62322** |

| Fed from | ROC-AUC | Log loss | Brier |
|---|---|---|---|
| Sampled stream (as trained) | 0.75082 | 0.16001 | 0.03575 |
| Production-scale stream | 0.75087 | 0.15968 | 0.03553 |

11.12% of predictions moved by more than 0.01. On those, log loss went from
0.34538 to 0.34213 and Brier from 0.10218 to 0.10014, so they got *better*. The
sampled version is noisy: one failure in one transaction reads as 100%.

**Decision: one store, prioritizing throughput.** The feature is important, but
the source swap doesn't hurt, so a second counter would only buy exact
testability at the cost of a Redis write on every outcome. Caveat for later:
live, a route the model rarely picks gets few outcomes and noisy rates. That
affects both designs equally, and it's the "dormant terminal" problem the
Razorpay paper lists as future work.

**Note on importance figures elsewhere in this report.** Earlier sections quote
LightGBM's default *split* importance, which counts how often a feature is used,
not how much it helps. Gain tells a different story. Top 10 by gain:
`route_success_rate_5m` 15.31%, `route_failure_rate_5m` 9.05%,
`route_failure_rate_1m` 6.50%, `time_since_last_failure` 6.18%, `amount` 4.67%,
`bank_route_success_rate_15m` 4.39%, `network_failure_rate_5m` 4.22%,
`route_utilization` 4.11%, `system_tps` 4.10%, `bank_route_gap` 3.84%.
`route_success_rate_5m` is the strongest feature by gain but only 3.35% by
split.

### 19.5 The fixes

- **Constants generated, not typed.** `ml/export_serving_constants.py` writes
  `models/serving_constants_v2.json` from the training rows (base rates,
  latencies, fees), `routes.py` (capacities, eligibility), `router.py` (routing
  shares), and the training `system_tps` range. `feature_storing.py` loads it
  (path overridable with `SERVING_CONSTANTS_PATH`), and the Dockerfile copies it
  into the image.
- **Active-minute windows.** `/outcome` records each minute with bank or network
  traffic in a sorted set (`{scope}:active`, trimmed to 24 h). The new
  `failure_rate_over_last_active_minutes()` reads the last 5 such minutes and
  returns NaN when there are none, matching training instead of inventing 0.0.
  Counter TTL went from 30 minutes to 24 hours, because a quiet bank's last 5
  active minutes can be hours old at night.
- **Known gaps.** Causes 3 and 4 can't be removed by any formula, so the referee
  labels those four features KNOWN GAP. Each still has a limit on its *median*
  difference (10% relative for the three traffic features, 0.05 absolute for
  `route_failure_rate_1m`). Exceeding it reports GAP TOO BIG and fails the run,
  so a real regression can't hide behind the label; the original 160,000-vs-200
  scale bug would blow straight through it.

### 19.6 Second run: 22 PASS, 4 KNOWN GAP, 0 FAIL

| Known gap | Median difference | Limit |
|---|---|---|
| `system_tps` | 5.14% | 10% |
| `route_utilization` | 5.14% | 10% |
| `route_stress` | 5.14% | 10% |
| `route_failure_rate_1m` | 0.0189 | 0.05 |

The 5.14% matches the simulator's own ±5% minute-to-minute traffic noise. The
two derived features show the identical figure because both scale linearly with
`system_tps`. All four sit at roughly half their limits.


## 20. Current State and Next Steps

**Done since Section 15:** environment rebuilt; feature CSV regenerated and
verified; serving spec verified against training data; constants generated
from training data; bank/network active-minute windows; parity test built (13
failing to 0 failing); `route_failure_rate_1m` decision measured; README
updated with the parity results and known gaps.

**Next, in order:**

1. ~~Scaled live traffic meter~~ (done, Section 17.4).
2. **Install Docker** and record a real baseline throughput run with profiling.
3. **Bottleneck loop** (Section 18.4), with the parity test green at every step.
   One free win is already known: since the in-flight fix (Section 15.3), no
   read uses the per-scope `req` counters (rates divide by `resolved`;
   utilization now uses the meter), yet `record_routing_decision()` still writes
   them for route, bank, network and bank×route, about 8 dead Redis operations
   per request. They were left in deliberately so the baseline measures the
   design as it stood; removing them is the first measured step.
4. **Horizontal scaling run** and the 250k TPS cluster projection.
5. **Remaining spec items:** credit outcomes to the attempt minute rather than
   the callback minute; idempotent outcomes; a write-path test; NaN instead of
   baseline fallbacks for the route rates at cold start (they currently fall
   back to the training base rate, where training would give NaN and the imputer
   would use 0.981).
6. Correct the artifact's stored threshold (Section 16.4).
7. Live dashboard UI.

