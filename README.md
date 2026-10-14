# FluxRoute: Smart Payment Routing

ML-driven payment gateway routing: given a transaction and several eligible
PSP routes, predict which route is most likely to succeed and route to it.
Modeled on the problem described in Razorpay's
[Smart Routing paper](https://arxiv.org/abs/2111.00783) (Bygari et al., 2021).

> **Data is synthetic.** No public labeled dataset exists for payment routing,
> so the dataset is simulated (see [Dataset Generation](#dataset-generation)).
> All metrics below are measured on that simulated data, not production traffic.

---

## Status

| Phase | State |
|---|---|
| Dataset generation (load-aware route health simulation) | Complete |
| Feature engineering (33 features, leakage-audited) | Complete |
| Model training, calibration, cost-based threshold selection | Complete |
| Offline routing evaluation vs. round-robin baseline | Complete |
| FastAPI serving layer + Redis feature store | Complete |
| Train/serve parity test (`tests/parity/`) | 22 exact, 4 documented known gaps, 0 failing |
| Write-path equivalence test (`tests/write_parity/`) | Passing — 1,320 Redis keys identical (type, value, TTL) |
| Containerization, 3 replicas behind nginx, Prometheus + Grafana | Complete |
| Load testing + throughput optimization loop | 6 measured runs: **139 → 210 TPS (+51%), 27.4 → 16.1 ms CPU per transaction (−41%)** |
| Scaling projection to 250k TPS ([`projection.md`](projection.md)) | Written — estimates labeled, with milestones to measure them |
| Per-core optimization queue (feature cache, pandas-free scoring, …) | Planned — [`deployment-report.md`](deployment-report.md) 9.14 |
| CacheX integration (custom Redis clone, separate repo) | Planned |

---

## Results

**Model** (LightGBM, 165,857-transaction held-out test week):

| Metric | Value |
|---|---|
| ROC-AUC | 0.751 |
| PR-AUC | 0.979 |
| Brier (pre-calibration) | 0.0358 |
| Brier (post-isotonic) | **0.0288** |
| Failure-class precision @ operating threshold | 0.774 |
| Failure-class recall @ operating threshold | 0.349 |
| Success-class precision | 0.971 |

**Routing** (offline evaluation, full test week):

| Policy | Success rate |
|---|---|
| Round-robin baseline | 95.44% |
| Weighted-random (the policy that generated the data) | 95.81% |
| **Model argmax** | **96.90%** |

**+1.47 percentage points over round-robin.** Razorpay's paper reports a 4-6%
lift from a month-long production A/B test; this is an offline estimate on
synthetic data with a simpler feature pipeline, so a smaller delta is expected.

**Benchmark comparison** — the paper publishes per-model-family metrics, which
makes a like-for-like check possible:

| | Razorpay LightGBM | Razorpay Random Forest (their production choice) | This project |
|---|---|---|---|
| Precision | 0.9433 | 0.9469 | 0.9714 |
| ROC-AUC | 0.7130 | 0.7949 | 0.7508 |

Comparable on the same model family — on synthetic data against their
production-scale system, so this is a sanity check on modeling approach, not a
claim of outperforming their system.

**Serving performance** (3 replicas on one 16-core laptop, Locust sweep at
10/25/50/100(/150) users; a transaction is one `/route` + one `/outcome`):

| Run | Change | Peak TPS | CPU ms per txn (u50) | p99 at u50 |
|---|---|---|---|---|
| baseline2 | instrumented starting point | 138.9 | 27.4 | 280 ms |
| A | remove Redis counters nothing read | 141.1 | 26.9 | 270 ms |
| B | `OMP_NUM_THREADS=1` | 140.0 | 27.1 | 280 ms |
| B2 | single-threaded LightGBM predict | 145.4 | 25.7 | 260 ms |
| C | pipelined feature reads (~80 → ~8 round trips) | 177.4 | 19.2 | 200 ms |
| **C2** | **pipelined writes (~42 → ~12 round trips per txn)** | **209.5** | **16.1** | **170 ms** |

One change per run, each measured with an identical procedure (flush, warm-up,
sweep, Prometheus queried at recorded timestamps) and gated by the parity tests.
What the measurements showed:

- **The bottleneck was never Redis** (it peaked at ~46% of one core). It was
  Python's GIL: each replica runs one process, capped at about one core, and
  ~80 sequential Redis round trips per request meant ~80 waits to re-acquire
  the GIL behind every other request thread.
- **Pipelining gained superlinearly.** Removing 8 round trips gained ~1.6%;
  removing ~72 gained up to +32%, because shorter requests mean fewer threads
  in flight, which means shorter waits for every remaining handoff.
- **A training setting leaked into serving.** The model was trained with
  `n_jobs=-1`, which LightGBM pickles and turns into an explicit 16-thread count
  on every predict — ~530 threads per replica under load, overriding
  `OMP_NUM_THREADS`. Fixed at load time with `set_params(n_jobs=1)`.
- **A round trip costs ~80–90 µs of app CPU even uncontended**, and CPU per
  transaction (from `process_cpu_seconds_total`) is a far more precise
  experiment metric than throughput: <1% run-to-run noise vs ±3%.
- **Past the knee (~u50), throughput falls while total CPU stays flat** — each
  extra concurrent request adds GIL contention, not work. Admission control at
  the knee is the next cheap fix.

Full write-up of every run, including predictions that failed and why:
[`deployment-report.md`](deployment-report.md) Part 9. What it would take to
reach 250k TPS: [`projection.md`](projection.md).

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Transaction    │────▶│  FastAPI         │────▶│  Chosen     │
│  request        │     │  /route          │     │  route      │
└─────────────────┘     └────────┬─────────┘     └─────────────┘
                                 │
                        reads 33 features
                        per candidate route
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Redis           │
                        │  feature store   │
                        │  (rolling stats) │
                        └────────▲─────────┘
                                 │
                        writes outcome
                                 │
                        ┌────────┴─────────┐
                        │  FastAPI         │
                        │  /outcome        │
                        └──────────────────┘
```

The service scores every eligible route for an incoming transaction and picks
the argmax — the same decision rule the Razorpay paper describes ("the payment
is processed through the terminal with the highest predicted probability of
success"). Outcomes are fed back via `/outcome`, updating the rolling route-health
statistics that the next request reads. That feedback loop is what makes route
health live rather than static.

### Repo layout

```
dataset-generator-routing/   # synthetic data + feature extraction
ml/                          # training, calibration, cost analysis
models/                      # serialized model + calibrator + serving constants
serving/                     # FastAPI service + Redis feature store + Dockerfile
deploy/                      # nginx, Prometheus, Grafana provisioning
loadtest/                    # Locust scenario + per-run result CSVs
tests/parity/                # train/serve feature parity (reads)
tests/write_parity/          # old-vs-new Redis write equivalence (writes)
report.md                    # ML engineering log
deployment-report.md         # serving, deployment, load tests, optimization loop
projection.md                # path from ~210 TPS to 250k+ TPS
```

---

## Dataset Generation

Four PSP routes, each with a base success rate, latency, fee, and **rated
capacity**. Route health follows a three-state process (HEALTHY / DEGRADED /
OUTAGE) driven by random incidents, and — critically — by **load**: a route
whose assigned traffic approaches its capacity degrades in both success rate and
latency, following a queueing-shaped curve.

To model production-scale congestion without a production-scale CSV, the
simulator separates two things:

- **Aggregate traffic** (~100k TPS) drives route utilization and health. Per
  route per minute this collapses into two integers via a binomial draw — never
  materialized as rows.
- **Sampled transactions** (~15/minute) are the actual ML training rows.

This keeps the dataset at ~660k rows while route-health statistics are computed
over realistic volume, which makes the rolling features far less noisy than
deriving them from the sampled rows alone.

28 simulated days, split chronologically 75/25 — **not** randomly. Route health
drifts over time, so a random split would leak future route state into training.

### Features (33: 26 numeric + 7 categorical)

- **Transaction**: amount, bank, network, payment method, merchant category,
  device, risk tier, hour-of-day, day-of-week
- **Route (static)**: base success rate, base latency, fee percentage
- **Route (live)**: 5m/15m rolling success rate, 5m rolling latency,
  utilization, system TPS, time since last degradation, circuit-breaker flag
- **Interaction**: bank×route success rate (Bayesian-shrunk toward the route
  baseline to stabilize low-volume pairs), bank×route gap, bank and network
  failure rates, amount vs. bank's historical average
- **Derived**: failure rates, success drop vs. baseline, latency ratio, route
  stress (utilization × failure rate)

Every rolling feature excludes the current minute by construction, so a
transaction's own outcome can never enter its own features.

---

## Engineering Notes

The interesting parts of this project were the bugs. Full detail in
[`report.md`](report.md); the ones that mattered most:

**Ground-truth leakage in a "real-time" feature.** `route_utilization` was
computed as load ÷ *health-adjusted* capacity — but health state is something
the simulator knows and a production service never would. Worse, since capacity
shrinks during incidents, the feature partly encoded the very thing it was
predicting. Found by asking "could a service actually compute this from Redis?",
not by any metric looking wrong. Redefining it against static rated capacity
cost nothing measurable (cost/txn and F1 unchanged) and made the model honestly
deployable.

**A feature that would have silently died at scale.** `time_since_last_failure`
was defined as "minutes since any failure." Fine at sampled volume; meaningless
at aggregate volume, where some failure occurs almost every minute by sheer
count. Caught by reasoning about the definition before running, then redefined
against an observable signal (rolling success rate crossing the circuit-breaker
threshold). It's now a top-3 feature by importance.

**A cost function that recommended rejecting everything.** The first version
counted PSP fees as a cost on successful transactions rather than revenue,
making "predict every transaction as a failure" optimal. The tell was that the
optimum landed on a sweep *boundary* rather than an interior point — a boundary
optimum is usually a modeling error, not a finding.

**Threshold selection by recall floor is fragile.** Selecting "cheapest
threshold achieving ≥50% failure recall" was compared against unconstrained cost
minimization across several model versions. The constrained choice repeatedly
landed in a catastrophic corner of the precision-recall curve (4-5% precision,
net financial loss) as the curve shifted between versions, while pure cost
minimization stayed stable. Threshold 0.16 was cost-optimal across three
consecutive model versions.

**A network-name typo that locked a whole segment to one route.** Two routes
declared `"RuPay"` while the generator emitted `"Rupay"`. Since eligibility is
exact string matching, every RuPay transaction was forced onto the single route
that happened to match — which was also the worst-performing one. Fixing it
widened the measured routing lift.

**A regeneration that silently didn't happen.** After a fix, retraining produced
row counts *identical to the previous run* — impossible from an unseeded Poisson
process over 40,320 minutes. The regenerated files had been written to the wrong
directory, so training had quietly reused stale data. Row-count forensics caught
what a clean exit code did not.

### Threshold and cost analysis

The operating threshold (0.16) minimizes a business cost function: missed
failures cost a fixed amount plus a percentage of the transaction, false alarms
cost the forgone PSP fee, and successful transactions earn it.

A sensitivity sweep over the cost assumptions (0.5×–10× the assumed failure
cost) showed recall is genuinely capped by model discriminative power, not by
conservative assumptions — even a 10× more pessimistic failure cost only pushes
recall to ~48%, and the system goes net-unprofitable at that point. An
independent hyperparameter re-search optimizing directly for cost (rather than
log loss) converged to the same operating point rather than beating it.

### Considered and rejected

**AdaCSL** ([Volk & Singer, 2021](https://arxiv.org/abs/2111.07382)) —
adaptive cost-sensitive learning that reweights loss based on per-probability-bin
calibration gaps. The paper is NN-specific but the theory is model-agnostic and
portable to boosted trees via sample weights and warm-started rounds. Rejected
after measuring: isotonic regression already closed the calibration gap it would
have targeted (Brier 0.0358 → 0.0288), so the added complexity had nothing left
to fix.

---

## Deployment

```
                    ┌──────────────────┐
   load  ─────────▶ │  nginx :8080     │  least_conn, keepalive
                    └────────┬─────────┘
                     ┌───────┼───────┐
                     ▼       ▼       ▼
                  ┌─────┐ ┌─────┐ ┌─────┐
                  │app1 │ │app2 │ │app3 │  uvicorn, 1 worker each
                  └──┬──┘ └──┬──┘ └──┬──┘
                     └───────┼───────┘
                             ▼
                     ┌───────────────┐        ┌────────────┐
                     │  Redis        │        │ Prometheus │──▶ Grafana
                     │ feature store │        │  :9090     │     :3000
                     └───────────────┘        └────────────┘
```

The service is stateless — all shared state lives in Redis, so replicas scale
horizontally without coordination. Each instance exposes `/metrics`; Prometheus
scrapes all three and Grafana provisions a dashboard covering p50/p95/p99
routing latency, per-instance throughput, route-selection distribution, observed
success rate, and a **latency breakdown separating Redis feature-fetch time from
model scoring time** — which is the number that tells you whether the cache or
the model is the bottleneck, and the baseline the CacheX comparison will be
measured against.

```bash
docker compose up --build

curl http://localhost:8080/health
```

| Service | URL |
|---|---|
| Routing API (via nginx) | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana (anonymous viewer) | http://localhost:3000 |

### Load testing

```bash
pip install -r loadtest/requirements.txt
locust -f loadtest/locustfile.py --host http://localhost:8080
```

The Locust scenario samples transactions from the same distributions as the
dataset generator, calls `/route`, then reports a simulated outcome back to
`/outcome` — so the load test drives the real feedback loop rather than
hammering a read-only endpoint, and route health actually evolves under load.

Recorded runs use a fixed procedure — `FLUSHALL`, a 2-minute warm-up, then a
headless sweep at 10/25/50/100/150 users, 3 minutes each, with every level's
end time logged so Prometheus percentiles and CPU per transaction are queried
afterwards at exact timestamps. Copy-paste commands are in
[`deployment-report.md`](deployment-report.md) Appendix A; results are in
[Results](#results) above.

### Live traffic meter

`system_tps` is the model's "how busy is the whole payment system" input. It was
trained on the simulator's production-scale stream (8,458 to 308,555 TPS), but a
live service can only count the requests that reach it, a few hundred per second
under a load test. Fed raw, a tree model treats that as the quietest night it
ever saw, on every request.

The service treats its own traffic as a sample of a larger stream, the way the
dataset itself is a 1-in-400,000 sample:

1. Every `/route` request increments a per-second counter in Redis.
2. Each request reads the last 60 *completed* seconds (so it never counts itself)
   once, and all candidate routes share that reading.
3. The rate is multiplied by `TRAFFIC_SCALE`: how many requests in the full
   system each request represents.
4. The result is clamped to the training range.
5. `route_utilization = system_tps × routing_share / capacity`, using the
   training's fixed routing shares. Using each route's *actual* traffic would let
   the feature describe the model's own routing choices, which training never
   showed it.

Choose `TRAFFIC_SCALE` so the planned load lands mid-range:
`162,978 / planned req/s` (about 800 for 200 req/s; the compose default).
Override per run with `TRAFFIC_SCALE=500 docker compose up --build`. Grafana can
show the reading (`fluxroute_system_tps`) and how often it hits the clamp
(`fluxroute_system_tps_clamped_total`). Constant clamping means `TRAFFIC_SCALE`
is wrong for the load being run. `tests/traffic_meter_check.py` checks the
scaling, both clamps, and that the current second is excluded.

This changes what the model decides, not how long deciding takes, so it doesn't
affect throughput measurements. Remaining limitation: in training, high traffic
always meant evening, so a flat load test can produce combinations the model
rarely saw (peak traffic at 3am). Retraining with the live-meter definition is
the long-term fix.

---

## Train/Serve Parity

The 26 numeric features are computed twice: in batch by the training pipeline
(pandas over the full history) and live by the serving layer (Redis, one request
at a time). If the two disagree, nothing crashes. The model just silently
receives inputs that mean something different from what it learned
(*training-serving skew*).

`tests/parity/` checks this directly. It takes 500 transactions from the test
week (ordinary moments, moments during route incidents, and quiet night hours),
rebuilds the exact history that existed just before each one, runs the serving
feature code on it, and compares every feature against what training computed.

**Current result: 22 PASS, 4 KNOWN GAP, 0 FAIL.**

The first run against the original serving code failed 13 of 26 features, from
four root causes:

| Root cause | Features | Outcome |
|---|---|---|
| Route constants typed in by hand from the simulator's nominal settings, not what training measured (R4 latency 175 ms vs. 283.69 ms actual) | 7 (one bug spreading into derived features) | **Fixed.** Constants are now generated from the training data by `ml/export_serving_constants.py` into `models/serving_constants_v2.json` |
| Bank/network failure rates used the last 5 clock minutes; training used the last 5 minutes that had any traffic | 2 | **Fixed.** Active minutes tracked in a Redis sorted set |
| Training knew the current minute's traffic; a live service only knows the previous minute's | 3 | Known gap |
| Training built one feature from a different data stream than the rest | 1 | Known gap |

### Known gaps

These cannot match exactly. They are still measured on every run, and each has a
limit on its typical (median) difference, so a real regression can't hide behind
the label.

| Feature | Median difference | Allowed | Why |
|---|---|---|---|
| `system_tps` | 5.1% | 10% | The simulator decides each minute's traffic at the start of the minute, so training knew it. A live service can only measure traffic that has already arrived and uses the previous minute. The 5% matches the simulator's own minute-to-minute traffic noise. |
| `route_utilization` | 5.1% | 10% | Built from `system_tps`; inherits the same gap. |
| `route_stress` | 5.1% | 10% | Built from `route_utilization`; inherits the same gap. |
| `route_failure_rate_1m` | 0.019 (absolute) | 0.05 | Training computed it from the sampled transactions (a handful per route per minute); the other route-health features came from the production-scale stream. A live service has one stream: its own outcomes. |

**Why `route_failure_rate_1m` was left as a gap rather than given its own
counter.** The feature matters: by gain it ranks 3rd of 33 inputs, and shuffling
it drops ROC-AUC from 0.751 to 0.623. But feeding it from the production-scale
stream instead costs nothing:

| | As trained (sampled) | Production-scale stream |
|---|---|---|
| ROC-AUC | 0.75082 | 0.75087 |
| Log loss | 0.16001 | 0.15968 |
| Brier | 0.03575 | 0.03553 |

The 11% of predictions that moved got slightly *better*. The sampled version is
noisy: one failure out of one transaction reads as 100%. A separate counter would
have added a Redis write to every outcome to buy exact testability, not better
predictions, so throughput won.

**What the parity test does not cover.** It feeds the serving code the same data
training used, so it verifies *formulas*, not *inputs*. Live, the service
measures its own traffic, which is orders of magnitude below the ~160k TPS
production stream `system_tps` was trained on. That's a separate problem, handled
by the scaled traffic meter (see [Live traffic meter](#live-traffic-meter)). It also doesn't exercise the write
path (how `/outcome` updates Redis); that has its own test.

### Write-path equivalence

`tests/write_parity/` covers the writes. It keeps a frozen copy of the previous
write code as a referee, replays one seeded stream of ~1,600 routing decisions
and ~1,600 outcomes through both implementations against separate in-memory
Redis instances, and requires every key's type, value, and TTL to match at every
simulated minute. The stream deliberately reaches every branch — failures, a
route outage that triggers the flag-down write, late outcomes, duplicate and
unknown outcome reports, decisions that never resolve — and the test refuses to
pass if any branch went unexercised. It gated the pipelined-writes change
(1,320 keys identical).

---

## Running it

```bash
pip install -r requirements.txt

# Generate dataset (~28 simulated days)
cd dataset-generator-routing && python generate.py && cd ..

# Train, calibrate, evaluate
python ml/lightgbm_tuned.py
python ml/isotonic_recalbration.py
python ml/combo_cost_metrics.py
python ml/sensitivity_check.py
python routing_simulation.py

# Serving constants (generated from training data) and the parity test
python ml/export_serving_constants.py
pip install -r tests/requirements.txt
python tests/parity/run_parity.py
python tests/write_parity/run_write_parity.py
python tests/traffic_meter_check.py

# Serve (full stack)
docker compose up --build
```

```bash
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{"amount": 2500.0, "bank": "HDFC", "network": "UPI",
       "payment_method": "UPI", "merchant": "Shopping",
       "device": "Android", "risk": "LOW"}'
```

For a single local instance without Docker:

```bash
docker run -d -p 6379:6379 redis
cd serving && uvicorn main:app --reload
```

---

## In Progress / Next

- **Per-core optimization queue** — profile first (py-spy, E0), then a
  per-minute in-process feature cache (exact, because every window excludes the
  current minute), pandas-free scoring, compiled tree inference, local write
  aggregation, and admission control at the throughput knee. Each one measured
  as CPU per transaction with the same runbook. Plan and risks:
  [`deployment-report.md`](deployment-report.md) 9.14–9.15.
- **Scaling to 250k TPS** — [`projection.md`](projection.md) works it through:
  the current design can't get there at any hardware budget (hot Redis keys),
  the data-access redesign that removes that ceiling, and ~500–800 vCPU
  (optimized Python) or under 100 (compiled hot path) after it — with the
  milestones that turn each estimate into a measurement.
- **Keepalive 502 fix** — a rare nginx/uvicorn keepalive race drops ~1 in
  100k+ requests; fix is uvicorn's keepalive timeout outliving nginx's upstream
  keepalive. Deferred so the optimization runs stayed single-variable.
- **More model families** — compare other models against the current LightGBM
  on the same test week and routing simulation, with serving inference cost as
  an explicit acceptance criterion alongside ROC-AUC and routing lift.
- **CacheX swap-in** — replacing Redis with a from-scratch C++ RESP2-compatible
  cache (separate repo), then re-running the identical Locust suite to measure
  the overhead of a hand-written cache vs. production Redis. The Grafana panel
  separating feature-fetch latency from scoring latency exists specifically to
  isolate this.
- **Closed-loop routing evaluation** — the offline simulation assumes a single
  routing decision doesn't affect future route health. Now that load drives
  degradation, a sequential evaluation where decisions feed back into
  utilization would be more rigorous. (The deployed service is already a real
  closed loop, so load testing measures this directly.)
- **Cost-aware ranking** — routing currently maximizes success probability, not
  expected profit. Razorpay's paper lists the same thing as future work.
- **Timezone anchoring** — `hour_of_day` is derived from UTC at serving time,
  while the simulator's traffic curve is in unanchored "local" time. Harmless on
  synthetic data, but a real deployment would need the serving clock aligned to
  whatever timezone the traffic pattern was learned in.
