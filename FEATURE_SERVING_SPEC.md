# Phase 1: live feature-serving contract

**Scope.** This is the serving contract for `models/lightgbm_tuned_v2.joblib`.
The model has 26 numeric inputs (specified below) plus seven categorical inputs:
`bank`, `network`, `payment_method`, `merchant`, `device`, `risk`, and
`route_id`.  A candidate route is scored with the same transaction categories
and a different `route_id`/route-specific feature vector.

This document intentionally describes the *code's actual behavior*, not just
the comments around it.  The authoritative offline sources are
`dataset-generator-routing/feature_extraction3.py` and
`ml/lightgbm_tuned.py::engineer_features()`.

## Non-negotiable serving rules

1. All minute-window features are read before the candidate request is
   recorded.  An outcome only changes a later prediction.
2. `m = floor(UTC event timestamp / 60)` is the common minute index.  The
   route outcome must retain its **attempt minute**, so a late callback is
   credited to that original minute rather than the callback's completion
   minute.
3. A route's aggregate health data must cover all production attempts, not
   only requests sampled for logging or model scoring.
4. Return `NaN`, rather than a made-up zero, wherever the offline formula has
   no denominator/history.  The saved numeric preprocessor imputes these with
   its fitted training median.
5. Feature reads and feature-state transitions must be atomic (one Redis Lua
   script or a transaction).  In particular, `time_since_last_failure` is a
   minute-level state transition, not an outcome-update field.

## Redis keyspace and lifecycle

`{r}` is route ID, `{b}` bank, `{n}` network, and `{m}` UTC minute.

| Key | Type / fields | Written when | Retention |
|---|---|---|---|
| `fr:route:{r}:m:{m}` | hash: `attempts`, `successes`, `latency_sum_ms` | outcome recording; increment `attempts`, increment successes for success=1, add observed latency | at least 16 completed minutes; recommend 24 h |
| `fr:bank-route:{b}:{r}:m:{m}` | same three fields | same outcome event | at least 16 completed minutes |
| `fr:bank:{b}:m:{m}` | hash: `attempts`, `failures`, `amount_sum`, `amount_count` | outcome adds attempt/failure; accepted request adds amount/count | at least 6 active-minute records plus amount history policy |
| `fr:network:{n}:m:{m}` | hash: `attempts`, `failures` | outcome event | at least 6 active-minute records |
| `fr:route-active:{r}`, `fr:bank-active:{b}`, `fr:network-active:{n}` | sorted sets, member=`m`, score=`m` | add member when that aggregate receives its first event in minute `m` | trim alongside minute hashes |
| `fr:bank-route-last-success:{b}:{r}` | hash: `minute`, `sequence` | successful outcome | model lifetime / durable |
| `fr:bank-amount:{b}` | hash: `sum`, `count` | accepted request, after its prediction is assembled | model lifetime / durable |
| `fr:route-incident:{r}` | hash: `last_flagged_minute`, `last_evaluated_minute` | first feature read for each route/minute | model lifetime / durable |
| `fr:static:route:{r}` | hash: frozen model-bound route constants | deployment bootstrap only | no expiry |
| `fr:ingress:sec:{s}` | integer: all accepted requests in second `s` | ingress, before feature read | 61 seconds |

Keep amount statistics separate from outcome aggregates because `amount` is
known at acceptance time and the offline feature is based on previous
transactions, regardless of whether their outcome has resolved.

Every outcome must be idempotent.  Use `fr:outcome:{attempt_id}` (`SET ... NX`
with expiry) inside the update script before any counter increments; retries
must not double count a failure or latency.

## Feature map (26 numeric inputs)

| # | Feature | Redis source / update rule | Read-time formula, exactly matching offline logic |
|---:|---|---|---|
| 1 | `day_in_week` | none | `(m // 1440) % 7`. The epoch/weekday anchor must be fixed to the one used for training; if production uses real calendar weeks, retraining is required. |
| 2 | `hour_of_day` | none | `(m % 1440) // 60`. |
| 3 | `amount` | request field | Input amount, unchanged. |
| 4 | `route_base_success_rate` | `fr:static:route:{r}.base_success_rate` | Frozen training-only mean for route `r`; never adapt it online. Current v2 values: R1 .9701643031658416, R2 .9561236269502552, R3 .96274718384772, R4 .944500315737774. |
| 5 | `route_base_latency_ms` | `fr:static:route:{r}.base_latency_ms` | Frozen training-only mean: R1 168.53432462479154, R2 116.1170768907857, R3 130.18236465209336, R4 283.69247035573125. |
| 6 | `route_cost_percent` | `fr:static:route:{r}.cost_percent` | Frozen config: R1 2.10, R2 1.45, R3 1.10, R4 1.75. |
| 7 | `route_success_rate_5m` | `fr:route:{r}:m:{m-5..m-1}` | `sum(successes) / sum(attempts)` over the five *clock minutes* `[m-5,m)`. Missing minute hashes contribute 0. If denominator is 0, `NaN`. |
| 8 | `route_success_rate_15m` | same | `sum(successes) / sum(attempts)` over `[m-15,m)`; `NaN` if denominator 0. |
| 9 | `route_avg_latency_5m` | same | `sum(latency_sum_ms) / sum(attempts)` over `[m-5,m)`; `NaN` if denominator 0. |
| 10 | `route_utilization` | `fr:ingress:sec:*` + static `routing_share`, `base_capacity_tps` | First calculate `system_tps` as feature 11. Then `system_tps * routing_share[r] / base_capacity_tps[r]`. Freeze shares to the training generator: R1 .35, R2 .25, R3 .25, R4 .15; capacities: 120000, 150000, 100000, 75000 TPS. Do **not** divide by an effective/health-reduced capacity and do not substitute selected-route traffic. |
| 11 | `system_tps` | `fr:ingress:sec:{s}` | At feature read, use the pre-request total ingress rate meter: `sum(requests in completed seconds s-60..s-1) / 60`. This is the live observable analogue of `generate_system_tps(minute)`, and must be the exact same value for every candidate route of a request. |
| 12 | `time_since_last_failure` | `fr:route-incident:{r}` plus feature 7 result | Return `m - last_flagged_minute` if one exists, else `9999.0`. Then, once per route/minute, if `route_success_rate_5m < 0.70` (and it is not NaN), set `last_flagged_minute=m`; otherwise leave it unchanged. The current minute's flag affects only later minutes. This is an incident-threshold feature, **not** “time since any failed transaction.” |
| 13 | `route_flagged_down` | feature 7 result | `1.0` iff non-NaN `route_success_rate_5m < 0.70`; otherwise `0.0`. |
| 14 | `bank_route_success_rate_15m` | `fr:bank-route:{b}:{r}:m:{m-15..m-1}` + static route baseline | With `A=sum(attempts)` and `S=sum(successes)` over `[m-15,m)`, `(S + 10 * route_base_success_rate) / (A + 10)`. It is never NaN: a cold pair equals its route baseline. |
| 15 | `route_failure_rate_5m` | derived only | `1.0 - route_success_rate_5m` (therefore NaN when feature 7 is NaN). |
| 16 | `route_failure_rate_15m` | derived only | `1.0 - route_success_rate_15m`. |
| 17 | `route_success_drop_5m` | derived only | `route_base_success_rate - route_success_rate_5m`. |
| 18 | `route_success_drop_15m` | derived only | `route_base_success_rate - route_success_rate_15m`. |
| 19 | `route_latency_ratio` | derived only | `route_avg_latency_5m / (route_base_latency_ms + 1e-6)`. |
| 20 | `bank_route_gap` | derived only | `bank_route_success_rate_15m - route_success_rate_15m`. |
| 21 | `route_stress` | derived only | `route_utilization * route_failure_rate_15m`. It is deliberately utilization, not `route_current_load`. |
| 22 | `bank_failure_rate_5m` | `fr:bank-active:{b}` and corresponding `fr:bank:{b}:m:*` | Literal `engineer_features()` parity: choose the five most recent **active bank-minute records** with minute `< m`, not necessarily the five clock minutes; return `sum(failures)/sum(attempts)`, or NaN with no prior record. This preserves pandas `groupby().shift(1).rolling(5)` on its sparse `bank_minute` table. |
| 23 | `network_failure_rate_5m` | `fr:network-active:{n}` and hashes | Same literal sparse-record rule as feature 22, scoped to network: previous five active network-minute records, `sum(failures)/sum(attempts)`, else NaN. |
| 24 | `route_failure_rate_1m` | `fr:route-active:{r}` and hash | Literal offline parity: select the latest active route-minute record with minute `< m`; return its `failures / attempts`, else NaN. Despite its name, this is the prior *observed route row*, not necessarily clock minute `m-1`, because `engineer_features()` did not densify this table. |
| 25 | `time_since_last_bank_route_success` | `fr:bank-route-last-success:{b}:{r}` | `m - stored_minute`, else `9999.0`. On a successful outcome, set the stored minute after all reads for that outcome's prediction. Offline rows are ordered by `txn_id`; live parity requires a monotonically assigned request sequence and ordered outcome application for each `(bank, route)`. |
| 26 | `amount_to_bank_avg_ratio` | `fr:bank-amount:{b}` | Before recording the current request amount, return `amount / (sum / count)` when `count > 0`; otherwise NaN. Then atomically increment `sum += amount`, `count += 1`. This is all prior bank transactions, with no time window. |

## Candidate-read and event-update order

1. Ingress assigns `attempt_id`, sequence, and minute `m`; increments the
   previous-second rate meter only after the feature snapshot, so the candidate
   does not see itself.
2. A single atomic feature-read script obtains the static values and all
   historical values above, calculates derived features 15–21, advances the
   per-route incident state once for `m`, and returns one vector per eligible
   candidate route. It must use a single `system_tps` snapshot for all
   candidates.
3. After the snapshot is committed, record the request in the ingress meter
   and bank amount accumulator. Route-attempt counters are incremented only
   when a route is actually selected—not once per candidate.
4. When the gateway resolves the selected attempt, the idempotent outcome
   script updates the route, bank-route, bank, and network aggregate hashes;
   active-minute sorted sets; and, for success, the bank-route success marker.

## Required compatibility decisions before Phase 2

Two differences are already present in the offline code and must not be
silently “fixed” in the service:

* `feature_extraction3.py` makes route and bank-route timelines dense, so
  features 7–9 and 14 are true clock-minute windows. In contrast, the three
  `engineer_features()` aggregations (features 22–24) are sparse and roll over
  previous populated rows. The table above specifies literal parity. Changing
  them to clock windows is a valid improvement only with regenerated features
  and retraining.
* Offline `system_tps` is known at the start of a simulated minute. A real
  service cannot know a minute's final throughput before making its first
  routing decision. The specified 60-second pre-request meter is observable
  and causal, but is not numerically identical to the simulator's exogenous
  draw. For strict distributional parity, feed a precomputed external traffic
  forecast into `system_tps` instead; otherwise retrain/evaluate with the live
  meter definition before production rollout.

`route_effective_capacity_tps`, `route_current_load`, and
`route_requests_1m` are present in the CSV but are **not model inputs** of v2.
Do not add them to the serving vector.  In particular, effective capacity was
health-state-derived in the simulator and is not an observable serving input.

## Review of current workspace changes

The intended v2 change set is coherent: aggregate route metrics were added,
`feature_extraction3.py` moved route health windows to that aggregate traffic,
and the tuned model now uses observable `route_utilization` and a thresholded
`time_since_last_failure`.  The model correctly omits `minute`,
`route_current_load`, `route_requests_1m`, and effective capacity.

The two compatibility traps above are the remaining high-risk train/serve
items.  A third operational constraint is outcome ordering: delayed gateway
callbacks must carry their original attempt minute and sequence, otherwise the
historical features can be corrupted even if all Redis counters are correct.

This workspace's `.git` directory is empty, so it is not a usable Git
repository in the current environment.  I could inspect the files and their
timestamps/differences between versioned filenames, but cannot provide `git
status`, a diff against HEAD, or safely run a Git checkout/revert here.
