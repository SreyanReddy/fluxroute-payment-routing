# FluxRoute — Smart Payment Routing

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
| Feature engineering (26 features, leakage-audited) | Complete |
| Model training, calibration, cost-based threshold selection | Complete |
| Offline routing evaluation vs. round-robin baseline | Complete |
| FastAPI serving layer + Redis feature store | Working locally |
| Containerization, replicas, load testing, monitoring | **In progress** |
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

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Transaction    │────▶│  FastAPI         │────▶│  Chosen     │
│  request        │     │  /route          │     │  route      │
└─────────────────┘     └────────┬─────────┘     └─────────────┘
                                 │
                        reads 26 features
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
models/                      # serialized model + calibrator
serving/                     # FastAPI service + Redis feature store
report.md                    # full engineering log
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

### Features (26)

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

## Running it

```bash
pip install -r requirements.txt

# Generate dataset (~28 simulated days)
cd dataset-generator-routing && python generate.py && cd ..

# Train, calibrate, evaluate
python ml/lightgbm_tuned.py
python ml/isotonic_recalibration.py
python ml/cost_function_check.py
python routing_simulation.py

# Serve
docker run -d -p 6379:6379 redis
cd serving && uvicorn main:app --reload
```

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"amount": 2500.0, "bank": "HDFC", "network": "UPI",
       "payment_method": "UPI", "merchant": "Shopping",
       "device": "Android", "risk": "LOW"}'
```

---

## In Progress / Next

- **Containerization and horizontal scaling** — Dockerfile and compose setup for
  the service + Redis, then multiple stateless replicas behind a load balancer
  sharing one Redis backend.
- **Load testing** — Locust/k6 against the replicated service for p99 latency
  and throughput under concurrency.
- **Monitoring** — Prometheus/Grafana, or structured logging plus a dashboard
  covering latency, route-selection distribution, and error rates.
- **CacheX swap-in** — replacing Redis with a from-scratch C++ RESP2-compatible
  cache (separate repo), then re-running the identical load-test suite to
  measure the overhead of a hand-written cache vs. production Redis.
- **Closed-loop routing evaluation** — the offline simulation assumes a single
  routing decision doesn't affect future route health. Now that load drives
  degradation, a sequential evaluation where decisions feed back into
  utilization would be more rigorous. (The deployed service is already a real
  closed loop, so load testing measures this directly.)
- **Cost-aware ranking** — routing currently maximizes success probability, not
  expected profit. Razorpay's paper lists the same thing as future work.
