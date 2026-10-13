# Deployment Report — What Changed and Why

A walkthrough of everything added to turn the trained model into a running,
observable, load-testable service. Written to be read start to finish; every
section explains the reasoning, not just the code.

Companion to `report.md` (which covers the ML pipeline). That document ends
where this one begins.

---

## Part 0 — The Mental Model

Before any code, understand what problem the serving layer actually solves,
because it is *not* the same problem the training pipeline solved.

**Training** had it easy. It had a CSV containing 28 days of history, all at
once. When it needed "R2's success rate over the last 15 minutes as of minute
30,000", it could look at rows 29,985 through 29,999, which were sitting right
there, already complete, with every outcome known.

**Serving** has none of that. A request arrives and asks "which route should
this transaction take?" and the service has to answer in milliseconds. There is
no CSV. There is no "last 15 minutes" sitting in a dataframe. The service has to
have been *accumulating* that information as transactions flowed through it, and
it has to be able to retrieve it instantly.

That is the entire job of the serving layer, and it splits into two halves:

1. **Remember**: after every transaction, write down what happened, in a shape
   that can be read back quickly.
2. **Reconstruct**: when a request arrives, read those notes back and rebuild
   the exact same 33 numbers the model was trained on.

Redis is the notebook. `serving/feature_storing.py` is both the writing and the
reading. `serving/main.py` is the thing that receives requests, asks
`feature_storing` for the numbers, runs the model, and answers.

The single most important property of this whole design is called **train/serve
parity**: the numbers reconstructed at serving time must be computed by *exactly*
the same formula as the numbers computed during training. If training computed
`route_success_rate_5m` one way and serving computes it a slightly different
way, the model receives inputs that look subtly wrong to it — and, critically,
**nothing crashes**. There is no error. The model just silently gets worse, and
you have no way to tell from the outside. This class of bug is the reason the
serving code mirrors the offline formulas so pedantically, and it is the reason
the biggest bug in this phase (Part 1) was worth catching.

---

## Part 1 — The Bug: In-Flight Transactions Were Counted as Failures

This is the most important change in the entire deployment phase. It is worth
understanding thoroughly, because it is a genuinely subtle bug and it is exactly
the kind of thing an interviewer will find interesting.

### 1.1 The transaction lifecycle

A payment transaction, in this system, has two separate moments in time:

```
   t=0ms                                          t=180ms
     |                                                |
     |  POST /route                                   |  POST /outcome
     |  "which route?"                                |  "it succeeded"
     |  -> service picks R2                           |  -> service records success
     |                                                |
     └──────────── transaction is "in flight" ────────┘
```

At `t=0` we know a request happened, and which route we chose. We do **not** yet
know whether it succeeded — the payment gateway hasn't responded. That answer
only arrives later, at `t=180ms`, via `/outcome`.

So the code has two counters, written at two different times:

- `req` — incremented at `t=0`, when the routing decision is made.
- `succ` — incremented at `t=180ms`, only if the transaction succeeded.

### 1.2 The bug

The original code computed success rate like this:

```python
successes = sum_window(scope, "succ", window_minutes)
requests  = sum_window(scope, "req",  window_minutes)
success_rate = successes / requests
```

That looks completely reasonable. It is wrong.

Consider a route that has handled 100 requests in the last 5 minutes. Suppose
90 of them have resolved, and all 90 succeeded — a perfect record. The other 10
are still in flight; their gateways haven't replied yet.

- `succ` = 90
- `req` = 100
- computed rate = 90 / 100 = **0.90**

But the true success rate is 90/90 = **1.00**. The route is perfect. The formula
reported it as 90% — because the 10 unresolved transactions were silently
treated as failures. They weren't failures. They just hadn't finished.

### 1.3 Why this gets worse under load, not better

This is the part that makes it dangerous rather than merely inaccurate.

The number of in-flight transactions at any instant is roughly
`throughput × average_latency` (this relationship is Little's Law). At 10
requests/second with 200ms latency, about 2 transactions are in flight at any
moment — a negligible distortion. At 500 requests/second with 200ms latency,
about 100 are in flight.

So the bias is small when the system is quiet and large when the system is busy.
Which means: **the feature is most wrong exactly when the system is under the
most load** — precisely when good routing decisions matter most. A bug that
hides during light testing and appears under production traffic is the worst
kind of bug to ship.

### 1.4 Why this is a train/serve parity violation

Offline, this bias does not exist at all. `feature_extraction3.py` works on
completed history, where every transaction in every minute bucket has a known
outcome. There is no such thing as "in flight" in a CSV.

So training computed a clean success rate, and serving computed a
downward-biased one. Same feature name, different meaning. That is the exact
failure mode described in Part 0 — the model receives inputs that don't match
what it learned from, and nothing crashes to tell you.

### 1.5 How it revealed itself

During manual testing, all four candidate routes started returning **identical**
predicted probabilities. That looked like a feature-computation bug ("nothing is
varying by route!"), but the real chain was:

1. Many `/route` calls were made; very few `/outcome` calls followed.
2. So `req` climbed for every route while `succ` stayed near zero.
3. Every route's computed success rate collapsed toward 0.
4. With every route's health features pinned to the same degraded values, the
   routes became nearly indistinguishable in feature space.
5. Their raw model scores landed very close together.
6. Close-together raw scores fell inside the **same isotonic calibration step**
   (see Part 2.4), which flattened them to bit-identical calibrated values.

A single root cause produced a symptom two layers away from it. Worth
remembering as a debugging lesson: the thing that looks broken is often not the
thing that *is* broken.

### 1.6 The fix

Add a third counter, `resolved`, incremented whenever an outcome arrives —
regardless of whether that outcome was success or failure:

```python
def record_outcome(scope, success, latency_ms=None):
    resolved_key = _bucket_key(scope, "resolved", minute)
    r.incr(resolved_key)
    r.expire(resolved_key, BUCKET_TTL_SECONDS)

    if success:
        # increment succ
    if latency_ms is not None:
        # increment lat
```

Then use `resolved` as the denominator for anything that is a *rate over
outcomes*:

```python
def get_rolling_rate(scope, window_minutes):
    resolved  = sum_window(scope, "resolved", window_minutes)
    successes = sum_window(scope, "succ", window_minutes)
    if resolved <= 0:
        return None, 0.0
    return successes / resolved, resolved
```

Now, in the earlier example: 90 successes ÷ 90 resolved = 1.00. Correct. The 10
in-flight transactions are invisible to the calculation until they resolve, at
which point they are counted properly as whatever they actually were.

### 1.7 Where `req` is still the right denominator

Crucially, `req` was **not** deleted, and this distinction matters:

| Feature | Denominator | Why |
|---|---|---|
| `route_success_rate_5m` | `resolved` | It is a rate *over outcomes*. An unresolved transaction has no outcome to average in. |
| `route_avg_latency_5m` | `resolved` | Latency is only recorded when the outcome arrives, so the numerator only contains resolved transactions. Denominator must match. |
| `bank_route_success_rate_15m` | `resolved` | Same reasoning as success rate. |
| `route_utilization` | `req` | This is *load* — how much traffic is hitting the route right now. An in-flight transaction is absolutely consuming capacity. It should count. |
| `system_tps` | `req` | Same: measuring arrival rate, not outcomes. |

Getting this split right is the actual insight. Both counters are correct; they
just answer different questions. "How busy is this route?" counts arrivals.
"How well is this route performing?" counts completions.

### 1.8 Files changed

`serving/feature_storing.py`, three edits:

1. `record_outcome()` — now increments `resolved` on every outcome.
2. `get_rolling_rate()` and `get_rolling_avg_latency()` — denominator switched
   from `req` to `resolved`.
3. `build_route_features()` — the bank×route shrunk rate switched from
   `sum_window(..., "req", ...)` to `sum_window(..., "resolved", ...)`.

---

## Part 2 — The Service (`serving/main.py`)

Rewritten. Four meaningful changes.

### 2.1 Feature list now comes from the model artifact

**Before:**

```python
sys.path.insert(0, os.path.join(THIS_DIR, "..", "ml"))
from lightgbm_tuned import NUMERIC_FEATURES, CATEGORICAL_FEATURES
```

**After:**

```python
_artifact = joblib.load(MODEL_PATH)
_feature_columns = _artifact["numeric_features"] + _artifact["categorical_features"]
```

Two independent reasons this is better.

**Reason one — correctness.** `save_model()` in `lightgbm_tuned.py` already
stores the feature lists inside the `.joblib` file alongside the model:

```python
artifact = {
    "model": model,
    "preprocessor": preprocessor,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
}
```

Those stored lists are a *snapshot of what this specific model was actually
fitted on*. If you later edit `NUMERIC_FEATURES` in `lightgbm_tuned.py` and
don't retrain, the file on disk and the deployed model disagree — and the old
import would have silently picked up the new, wrong list. Reading from the
artifact makes that class of drift structurally impossible: the features always
match the model, because they travel together in the same file.

**Reason two — deployment hygiene.** The serving container no longer needs the
`ml/` directory at all. It needs a model file and the serving code. Nothing
else. That is a smaller image, a smaller attack surface, and a cleaner
separation between "the thing that trains" and "the thing that serves" — which
in a real company are usually owned by different teams and deployed on entirely
different schedules.

### 2.2 Prometheus instrumentation

Five metrics were added. Understanding *why each one exists* matters more than
the code.

```python
ROUTE_REQUESTS = Counter(
    "fluxroute_route_requests_total", ..., ["instance", "status"]
)
```
A **Counter** only ever goes up. You never read a counter's raw value directly —
you read its *rate of change*, which gives throughput. The `status` label lets
you separate successful routing from `no_eligible_route` rejections, so you can
compute an error rate.

```python
ROUTE_LATENCY = Histogram(
    "fluxroute_route_latency_seconds", ..., buckets=(0.005, 0.01, 0.025, ...)
)
```
A **Histogram** is how you get percentiles. It does not store individual
timings — it stores counts per bucket ("how many requests took under 5ms? under
10ms? under 25ms?"). Prometheus then estimates p50/p95/p99 by interpolating
across buckets. The bucket boundaries were chosen to straddle the range this
service is expected to land in; buckets that are all far above or below your
actual latency give useless percentile estimates.

```python
SCORING_LATENCY = Histogram("fluxroute_scoring_latency_seconds", ...)
FEATURE_FETCH_LATENCY = Histogram("fluxroute_feature_fetch_seconds", ...)
```
**These two are the most valuable metrics in the whole system.** Total latency
tells you *that* the service is slow. These tell you *which half* is slow — the
Redis round-trips that fetch features, or the LightGBM model doing arithmetic.
Without this split, "p99 is 80ms" is a fact you can't act on. With it, "p99 is
80ms, of which 72ms is Redis" points directly at what to fix.

This split is also the exact instrument the CacheX comparison will need later:
swapping Redis for CacheX changes the feature-fetch number and leaves the
scoring number alone, so this is precisely where the difference will show up.

```python
ROUTE_SELECTED = Counter(..., ["instance", "route_id"])
OUTCOMES = Counter(..., ["instance", "route_id", "result"])
```
Business-level rather than performance-level. `ROUTE_SELECTED` shows whether the
model is actually discriminating between routes or funneling everything into
one. `OUTCOMES` gives the observed real-world success rate — the thing the model
is ultimately trying to maximize.

### 2.3 The `/metrics` endpoint

```python
@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Prometheus works on a **pull** model, which surprises people who expect push.
The application does not send metrics anywhere. It exposes a plain-text page,
and Prometheus visits that page on a schedule (every 5 seconds, per
`deploy/prometheus.yml`) and scrapes whatever it finds. Your app doesn't need to
know Prometheus exists or where it lives.

### 2.4 Ranking on raw probabilities, not calibrated ones

```python
candidates_df["raw_success_probability"] = raw_probabilities
candidates_df["predicted_success_probability"] = _calibrator.predict(raw_probabilities)

best_index = candidates_df["raw_success_probability"].idxmax()
chosen_probability = float(candidates_df.loc[best_index, "predicted_success_probability"])
```

Rank on **raw**, report **calibrated**. This deserves explanation because it
looks inconsistent at first glance.

Isotonic regression produces a **step function**. It is not a smooth curve — it
maps ranges of input onto single flat output values. Because ~95% of this data
sits above probability 0.9, the calibrator has relatively few distinct steps up
in that region, and each step is fairly wide.

Consequence: two routes with genuinely different raw scores — say 0.9984 and
0.9967 — can land inside the same step and come out with *bit-identical*
calibrated values. At which point `idxmax()` no longer picks the best route. It
picks whichever route happens to sit first in the DataFrame, which is
essentially R1 every time. Real information, silently discarded.

The resolution is to be clear about what each number is *for*:

- **Calibrated probability** answers "what is the true likelihood this
  succeeds?" Its value is that 0.97 actually means 97%. That is what you report
  to a caller, and what any downstream cost calculation needs.
- **Raw score** answers "which of these is better?" It has finer resolution
  because it hasn't been quantized into steps. That is what you rank on.

Calibration was built to make absolute probabilities meaningful. It was never
built to preserve fine-grained ordering between near-ties. Using each for its
actual purpose is not a compromise — it's the correct use of both.

### 2.5 Environment-variable configuration

```python
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(THIS_DIR, "..", "models", "..."))
INSTANCE_ID = os.environ.get("INSTANCE_ID", "local")
```

Same code runs locally (`models/` is one directory up) and in a container
(`/app/models/`) with no edits — the environment supplies the difference. This
is standard twelve-factor practice: configuration lives in the environment, not
in the code.

`INSTANCE_ID` is how you tell the three replicas apart in metrics and in
`/health` responses. It's also how you *prove* load balancing is working —
curl `/health` repeatedly and watch the instance name change.

---

## Part 3 — The Container

### 3.1 `serving/Dockerfile`, decision by decision

```dockerfile
FROM python:3.10-slim
```
**3.10 specifically**, because that is the Python your `.venv` uses and the
version the model was pickled under. joblib/pickle files are not guaranteed
portable across Python or scikit-learn versions — a mismatch can throw on load
or, worse, load with subtly different behavior. `slim` because the full image
carries hundreds of megabytes of build tooling this service never uses.

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
```
`libgomp1` is the GNU OpenMP runtime. **LightGBM will not import without it** —
it uses OpenMP for multithreading, and slim images don't ship it. This is the
single most common reason a LightGBM container fails to start.
`curl` is for the healthcheck. `rm -rf /var/lib/apt/lists/*` in the *same* layer
deletes the apt cache before the layer is committed — done in a later `RUN` it
would save nothing, because the files would already be baked into the earlier
layer.

```dockerfile
COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir -r /app/serving/requirements.txt

COPY models/... /app/models/...
COPY serving/ /app/serving/
```
**Order is deliberate.** Docker caches each layer and invalidates everything
after the first change. Requirements are copied and installed *before* the
source code, so editing `main.py` reuses the cached dependency layer and rebuilds
in seconds instead of reinstalling scikit-learn every time.

```dockerfile
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1
```
Docker now knows the difference between "the container process is running" and
"the application is actually able to serve". `--start-period=40s` is a grace
window — loading the model and calibrator takes a few seconds, and without it
the container would be marked unhealthy during normal startup.

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```
`--host 0.0.0.0` is mandatory in a container. The default `127.0.0.1` only
accepts connections originating *inside* the container, so nginx would never be
able to reach it.

**`--workers 1` is a deliberate correctness decision, not laziness.**
`prometheus_client` keeps its counters in ordinary process memory. With multiple
uvicorn workers, each worker is a separate OS process with its own separate
counters. Prometheus scrapes `:8000/metrics` once, gets routed by the OS to one
arbitrary worker, and receives only that worker's numbers — so with 2 workers
you'd see roughly half your traffic, fluctuating unpredictably as scrapes land
on different workers. Concurrency comes from running three replica *containers*
instead, which keeps each process's metrics exact and complete.

### 3.2 `.dockerignore`

Excludes `.venv/`, `__pycache__/`, all CSVs, `.git/`, and the docs. Without it,
Docker uploads the entire project directory to the build daemon as "build
context" — including a multi-hundred-megabyte virtualenv and every generated
dataset — before it runs a single instruction. This turns a slow build into a
fast one.

### 3.3 `serving/requirements.txt`

Separate from the root `requirements.txt` and deliberately smaller. The training
environment needs things the serving environment doesn't. Versions are pinned
exactly — particularly `scikit-learn==1.7.2` and `numpy==2.2.6`, because those
are what the model was pickled with, and unpickling under a different version is
where mysterious failures come from.

---

## Part 4 — The Stack (`docker-compose.yml`)

Six services in one file.

### 4.1 The YAML anchor

```yaml
x-app: &app
  build:
    context: .
    dockerfile: serving/Dockerfile
  depends_on:
    redis:
      condition: service_healthy
  restart: unless-stopped

services:
  app1:
    <<: *app
    environment:
      INSTANCE_ID: app1
```

`x-app` defines a reusable block (the `x-` prefix tells Compose to ignore it as
a service definition). `&app` names it; `<<: *app` merges it into each replica.
Three near-identical services without triplicating their config.

One subtlety worth knowing: YAML merge is **shallow**. When `app1` defines its
own `environment:`, it *replaces* the anchor's `environment:` entirely rather
than merging into it. That's why `REDIS_HOST` and `REDIS_PORT` are re-listed
explicitly in each service — omitting them would mean losing them.

```yaml
build:
  context: .
  dockerfile: serving/Dockerfile
```
Build context is the **project root**, not `serving/`. The Dockerfile needs to
`COPY models/...`, which lives outside `serving/` — and Docker cannot access
anything above its build context.

### 4.2 Redis

```yaml
redis:
  image: redis:7-alpine
  command: ["redis-server", "--maxmemory", "512mb", "--maxmemory-policy", "allkeys-lru"]
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
```

**`--maxmemory 512mb` with `allkeys-lru`** caps memory and evicts
least-recently-used keys when full, rather than the default behaviour of
refusing writes. Every key this service writes carries a TTL and is
reconstructible from continuing traffic, so eviction is a safe failure mode and
refusing writes is not.

The healthcheck matters because of `depends_on: condition: service_healthy` on
the app services. Plain `depends_on` only waits for a container to *start*,
which for Redis takes milliseconds but doesn't guarantee it's accepting
connections. With the condition, the apps genuinely wait until Redis answers
`PING`.

### 4.3 Port exposure

Only three ports are published to the host: `8080` (nginx), `9090`
(Prometheus), `3000` (Grafana). The three app containers are deliberately **not**
published — they're reachable only from inside the Docker network, which is
exactly right: all external traffic should arrive through the load balancer.

Redis publishes `6379` purely as a debugging convenience so you can run
`redis-cli` from your laptop. In a real deployment you'd remove that line.

---

## Part 5 — The Load Balancer (`deploy/nginx.conf`)

```nginx
upstream fluxroute_backend {
    least_conn;
    server app1:8000 max_fails=3 fail_timeout=10s;
    server app2:8000 max_fails=3 fail_timeout=10s;
    server app3:8000 max_fails=3 fail_timeout=10s;
    keepalive 128;
}
```

**`least_conn` rather than the default round-robin.** Round-robin distributes
requests evenly by count, which is optimal only when every request costs the
same. These don't: a UPI transaction is eligible for all four routes, a
Mastercard transaction for three — so requests do meaningfully different amounts
of work. `least_conn` sends each new request to whichever backend currently has
the fewest open connections, which naturally routes around an instance that's
bogged down.

**`max_fails=3 fail_timeout=10s`** is passive health checking: if a backend
fails 3 times within 10 seconds, nginx stops sending it traffic for 10 seconds,
then tentatively tries again. One sick replica degrades throughput instead of
returning errors to a third of users.

**`keepalive 128`** maintains a pool of persistent connections to the backends.
Without it nginx opens a fresh TCP connection per request and closes it after —
adding a full handshake to every single request. Under load this is a
significant, and completely avoidable, latency cost.

```nginx
proxy_http_version 1.1;
proxy_set_header Connection "";
```
These two lines are **required** for `keepalive` to actually work. HTTP/1.0
can't do persistent connections, and nginx's default `Connection: close` header
would tell the backend to hang up after each response, silently defeating the
connection pool. A very common misconfiguration — `keepalive` present but
inert.

```nginx
access_log off;
```
During load testing nginx would otherwise write a line to disk for every single
request, making disk I/O a component of your measured latency. You'd be
benchmarking your logger.

---

## Part 6 — Observability

### 6.1 `deploy/prometheus.yml`

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: fluxroute
    metrics_path: /metrics
    static_configs:
      - targets: [app1:8000, app2:8000, app3:8000]
```

Prometheus scrapes each app container directly, **bypassing nginx**. This is
intentional: going through the load balancer would land each scrape on one
arbitrary instance, so you'd get a random third of your metrics each time.
Scraping instances directly gives complete per-instance data that can then be
summed in queries.

`5s` is aggressive (production is typically 15-30s) but appropriate here — load
tests run for minutes, and a slow scrape interval would produce too few data
points to see anything.

Note the targets use **service names**, not IP addresses. Docker Compose runs an
internal DNS server that resolves service names to container IPs automatically.

### 6.2 Grafana provisioning

Two files under `deploy/grafana/`:

- `datasources/prometheus.yml` — tells Grafana where Prometheus lives, so you
  never have to click through the setup UI.
- `dashboards/dashboards.yml` + `dashboards/fluxroute.json` — the provider
  config and the dashboard itself.

The point of provisioning is that the dashboard is **code in the repository**,
not clicks someone made once in a browser. Anyone who clones the repo and runs
`docker compose up` gets the identical dashboard. Nothing to set up, nothing to
document, nothing to lose.

### 6.3 The dashboard panels

**Routing latency (p50/p95/p99)**
```
histogram_quantile(0.99, sum(rate(fluxroute_route_latency_seconds_bucket[1m])) by (le))
```
Read this inside-out, because this query pattern is worth actually
understanding:
- `rate(...[1m])` — per-second rate of increase of each bucket over a 1-minute
  window. Converts ever-increasing counters into a current rate.
- `sum(...) by (le)` — sums across all three instances, keeping the `le`
  ("less than or equal") bucket boundary label. This is what makes it a
  *fleet-wide* percentile rather than three separate ones.
- `histogram_quantile(0.99, ...)` — interpolates across the buckets to estimate
  the 99th percentile.

Watching all three of p50/p95/p99 together is far more informative than p99
alone. If p50 stays flat while p99 climbs, a small subset of requests is
suffering — usually queueing or contention. If all three rise together, the
whole system is saturated.

**Throughput by instance** — stacked per-instance request rate. The three bands
should be roughly equal; if one is consistently thinner, load balancing isn't
working as intended.

**Latency breakdown: feature fetch vs. scoring** — the diagnostic panel. Two
p99 lines on one axis, showing which half of the request path dominates.

**Route selection distribution** — are decisions spread across routes, or is
everything funneling into one? A flat line at 100% on a single route usually
means features aren't varying (exactly the symptom from Part 1.5).

**Observed success rate**
```
sum(rate(fluxroute_outcomes_total{result="success"}[5m]))
  / clamp_min(sum(rate(fluxroute_outcomes_total[5m])), 0.001)
```
`clamp_min` prevents a divide-by-zero producing `NaN` when there's no traffic —
a small robustness detail that keeps the panel from showing an error whenever
the system is idle.

---

## Part 7 — The Load Test (`loadtest/locustfile.py`)

### 7.1 Realistic input

The generator samples transactions from **the same distributions as
`transactions.py`** — the same bank weights (HDFC 30, ICICI 18, SBI 22, …), the
same log-normal amount distribution, the same amount-conditioned risk tiers, the
same CARD→Visa/Mastercard/Rupay and UPI→UPI network logic.

This matters more than it looks. Feeding uniform random inputs would exercise
route-eligibility paths in wrong proportions and produce a completely different
cache-hit and route-distribution profile than real traffic would. The load test
would be measuring a system that doesn't exist.

### 7.2 The closed loop — the important design decision

```python
with self.client.post("/route", ...) as response:
    body = response.json()
    transaction_id = body["transaction_id"]
    chosen_route_id = body["chosen_route_id"]

success, latency_ms = simulate_outcome(chosen_route_id)

self.client.post("/outcome", json={
    "transaction_id": transaction_id,
    "success": success,
    "latency_ms": latency_ms,
})
```

Each simulated user does **both halves** of the real lifecycle: asks for a
route, then reports back what happened.

A load test that only hammered `/route` would be nearly worthless here. Without
`/outcome` calls, `resolved` and `succ` never increment — route health features
stay frozen at their cold-start fallbacks forever, every route looks identical,
and you'd be benchmarking a degenerate version of the system that never occurs
in reality. (It would also have hidden the Part 1 bug entirely.)

Doing both halves means route health genuinely evolves during the test, features
actually vary, and the model is making real decisions on live data. Locust
reports `/route` and `/outcome` as separate named request types, so their
latencies stay cleanly separated in the results.

`simulate_outcome()` uses each route's real base success rate (R1 0.986, R2
0.981, R3 0.979, R4 0.983) so failures occur at a realistic rate rather than
everything succeeding.

---

## Part 8 — What to Expect on the First Real Run

### 8.1 A likely bottleneck, predicted in advance

Count the Redis round-trips in one `/route` call. Per candidate route,
`build_route_features()` performs roughly twenty separate operations —
`sum_window` calls for 5m/15m/1m success and latency windows, bank-scope and
network-scope windows, bank×route windows, two last-event lookups, two `GET`s
for the bank amount average. Multiply by four candidate routes:

**Roughly 80 sequential Redis round-trips per routing decision.**

They are sequential — each one waits for the previous to return before starting.
Even at a fast 0.2ms per local round-trip, that's ~16ms of pure network waiting
per request, and it will grow under concurrency as Redis queues requests.

**Prediction: `feature_fetch` will dominate `scoring` in p99, probably by a
large multiple.** LightGBM scoring four rows is genuinely fast — likely
sub-millisecond.

### 8.2 Why this is good news, not a problem

If that's what the data shows, you have:

1. A **measured** bottleneck rather than a guessed one.
2. An obvious, well-understood fix: Redis **pipelining** — batching all ~80
   operations into a single round-trip instead of 80. This is exactly what
   pipelining exists for, and it should cut p99 dramatically.
3. A genuine before/after result to report.

"I built it, measured it, found Redis round-trips dominated p99, batched them
into a pipeline, and cut p99 by X%" is a substantially better engineering story
than any single latency number, because it demonstrates the actual loop of
measure → diagnose → fix → verify.

**Get the baseline numbers first.** Optimizing before measuring throws away the
"before" half of the comparison, which is the half that proves you knew what you
were doing.

### 8.3 Why the warm-up step is not optional

On a cold Redis every rolling feature falls back to its static baseline, so all
routes look nearly identical and the model has nothing to discriminate on.
Additionally, the first few requests pay one-time costs (lazy imports, memory
allocation) that don't reflect steady-state behaviour.

Run 2 minutes of traffic before recording anything, then confirm the four
candidates in a single `/route` response now show **different** probabilities.
That is your direct evidence the Part 1 fix worked.

---

## Part 9 — What Is Still Not Done

Stated plainly so nothing in the repo overstates itself:

1. **No load-test run has been recorded.** The harness and the full stack exist
   and are runnable; the numbers do not exist until you run them. The README
   says exactly this, with a placeholder where they'll go.
2. **Redis pipelining** — see Part 8. Deliberately deferred until after
   baseline measurement.
3. **CacheX swap-in** — separate repo, not started. The feature-fetch vs.
   scoring latency split exists specifically so this comparison is measurable
   when it happens.
4. **Closed-loop offline evaluation** — the offline routing simulation still
   assumes one routing decision doesn't affect future route health. The
   *deployed* service is a genuine closed loop, so load testing measures this
   directly and the offline version matters less than it did.
5. **Cost-aware ranking** — `/route` maximizes success probability, not expected
   profit. This matches what Razorpay's paper describes for their production
   system, and their paper lists cost-aware routing as future work too.
6. **Timezone anchoring** — `hour_of_day` is derived from UTC at serving time,
   while the simulator's traffic curve is in unanchored "local" time. Harmless
   on synthetic data; a real deployment would need the serving clock aligned to
   whatever timezone the traffic pattern was learned in.
7. **No authentication, rate limiting, or TLS.** Appropriate for a portfolio
   project; worth naming as a known gap rather than pretending otherwise.

---

## Part 10 — Explaining This Out Loud

Compressed answers for the questions most likely to come up.

**"Walk me through what happens when a request hits your service."**
A transaction arrives at nginx, which forwards it to whichever of three replicas
has the fewest open connections. That instance determines which routes are
eligible for the payment method and network, then for each eligible route
reconstructs 33 features from Redis — rolling success rates, latency, capacity
utilization, bank-route interaction history. It scores all candidates through
LightGBM in a single batched call, takes the argmax on raw probability, records
the decision back to Redis, and returns the choice with a calibrated probability.
Later the caller reports the real outcome to `/outcome`, which updates the
rolling counters that the next request will read. That feedback loop is what
makes route health live rather than static.

**"What was the hardest bug?"**
Rolling success rate was computed as successes ÷ requests, but requests are
counted when the routing decision is made and successes only when the outcome
arrives — so every in-flight transaction was being counted as a failure. It's
subtle because it doesn't exist offline, where all history is complete, and it
gets *worse* under load, since in-flight count scales with throughput ×
latency. Fixed by tracking resolved outcomes separately and using that as the
denominator for rates, while keeping the request counter for utilization, which
genuinely should count in-flight work.

**"Why three replicas instead of more workers in one process?"**
Partly clean horizontal scaling — the service is stateless, all shared state is
in Redis, so replicas need no coordination. But specifically here, because
`prometheus_client` keeps counters in per-process memory: with multiple workers
behind one port, each scrape lands on an arbitrary worker and returns partial
data. One worker per container keeps metrics exact.

**"How do you know the model is actually helping?"**
Offline, routing by model argmax versus round-robin over a held-out test week:
96.9% versus 95.4% success, a 1.5-point lift. In the running service, the
Grafana route-distribution panel shows whether decisions are actually spread
across routes rather than collapsing onto one, and the observed success rate
panel tracks real outcomes.

**"What would you do next?"**
Measure, then pipeline the Redis reads — roughly 80 sequential round-trips per
request is almost certainly the p99 bottleneck, and batching them into one
round-trip is the obvious fix. After that, swap Redis for the C++ cache I'm
building and re-run the identical load test to measure the difference.
