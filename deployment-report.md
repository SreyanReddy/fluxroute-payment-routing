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

> **Later update (9.10):** the `req` counters were eventually removed. Arrivals
> are still what load features measure, but they now come from the traffic
> meter: `system_tps` counts per-second `fr:ingress` arrivals, and
> `route_utilization` is derived from `system_tps` × the route's fixed routing
> share. Once nothing read the per-minute `req` keys, writing them was pure
> overhead.

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

> **Outcome (Part 9):** half right. Fetch does dominate, but Redis sat at 39%
> CPU, so the cost was GIL contention around the round-trips in the app, not
> Redis itself. Scoring was 24 ms at p99, not sub-millisecond.

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

## Part 9 — The Baseline Run (Measured)

Part 8 was a prediction. This part is what actually happened when the stack ran
under load. The prediction turned out to be half right: feature fetch does
dominate, but the reason is not the one Part 8 gave.

### 9.1 Setup and procedure

| | |
|---|---|
| Code | commit `7a7d06f` |
| Host | ASUS Vivobook K6502VU, 16 logical cores, 15.2 GiB RAM, Zorin OS 17 (Ubuntu 22.04 base) |
| Runtime | Docker 29.9.0, Compose v5.6.0 |
| Stack | 3 app replicas × 1 uvicorn worker, nginx `least_conn`, Redis 7, Prometheus, Grafana |
| Load generator | Locust 2.32.4, same host as the stack |
| `TRAFFIC_SCALE` | 800 (compose default) |

Everything shares one laptop, so these numbers describe this machine, not a
production deployment. What carries over is the shape of the curve and the
diagnosis, not the absolute req/s.

The procedure, used for this run and to be repeated unchanged for every
after-run:

1. `redis-cli FLUSHALL` to start from a known state.
2. Warm up for 2 minutes at 50 users, without recording (see 8.3).
3. Sweep 10 → 25 → 50 → 100 users, 3 minutes each, back to back, without
   flushing between levels:
   ```bash
   locust -f loadtest/locustfile.py --host http://localhost:8080 --headless \
       -u <N> -r <min(N,25)> -t 3m --csv loadtest/baseline_u<N> --only-summary
   ```
4. Afterwards, evaluate the server-side Prometheus p99s **at each level's end
   time** (taken from Locust's shutdown log line) with a `[2m]` window, so the
   window covers the steady-state last 2 minutes and excludes ramp-up:
   ```bash
   curl -s localhost:9090/api/v1/query \
       --data-urlencode 'query=histogram_quantile(0.99, sum by (le) (rate(fluxroute_feature_fetch_seconds_bucket[2m])))' \
       --data-urlencode "time=$(date -d '2026-10-13 22:09:05' +%s)"
   ```

Copy-paste commands for the whole procedure are in **Appendix A**.

Step 4 is deliberate. The first attempt queried Prometheus by hand in the
browser between levels. Under time pressure, one query was pasted twice, which
made fetch and scoring p99 look identical at two load levels. Evaluating at
recorded timestamps after the fact removes that whole class of mistake.

Raw Locust output is in `loadtest/baseline_u{10,25,50,100}_stats.csv`.

### 9.2 Pre-flight checks

**An earlier run was discarded.** `loadtest/run1_*.csv`, recorded before the
stack existed in Docker, showed 53,631 requests at ~2 ms with **100%
failures**. That is the signature of nothing listening on the port. It
measured connection refusals, not the service, so it was deleted. The fix is
procedural: always smoke-test one `/route` call and confirm a 200 before
applying load.

**Warm-up works, and the model learns live.** The same request before and
after warm-up:

| Route | Simulator base success rate | Cold (empty Redis) | Warm (after 2 min) |
|---|---|---|---|
| R1 | 0.986 (best) | 0.9803 | **0.9790** (chosen) |
| R2 | 0.981 | 0.9790 | 0.9749 |
| R3 | 0.979 (worst) | 0.9790 | **0.9396** |
| R4 | 0.983 | 0.9803 | 0.9789 |

Cold, every route falls back to the same static defaults, so the candidates tie
exactly (R1 = R4, R2 = R3). The ties are exact rather than merely close because
isotonic calibration is a step function: slightly different raw scores land on
the same plateau. The service ranks on raw scores (2.4), which is why it can
still pick a winner among calibrated ties. Warm, the ordering recovers the
simulator's true ranking, best and worst at the ends, purely from outcomes
reported through `/outcome`. This is the live evidence that the Part 1 fix and
the feedback loop work.

**Load balancing is even under concurrent load.** Request counts per replica
after warm-up: app1 5,941 / app2 5,961 / app3 6,009, a spread of 1.2%.

A misleading first check is worth recording. Thirteen sequential `curl`s all
landed on app1 (13 / 0 / 0). That is not a balancing bug. nginx runs one worker
per core (`worker_processes auto`, 16 here), and each worker keeps its own
`least_conn` state. With sequential requests every backend has 0 active
connections, so every worker breaks the tie the same way, by picking the first
server. Only concurrent load creates the connection-count differences that
`least_conn` acts on.

**The procedure is reproducible.** Two separate 2-minute warm-ups at 50 users
gave 149.1 and 144.7 `/route` req/s, both with a 160 ms median. Two separate u10
runs gave 59.4 and 59.6 req/s with a 31 ms median. Locust never raised its
CPU-usage warning, so the load generator was not the limit.

### 9.3 Results

End-to-end, from Locust (client side, the source of truth for latency):

| Users | `/route` req/s | p50 | p95 | p99 | `/outcome` p50 | Failures |
|---|---|---|---|---|---|---|
| 10 | 59.6 | 31 ms | 58 ms | 76 ms | 7 ms | 0 |
| 25 | 128.4 | 51 ms | 96 ms | 130 ms | 13 ms | 0 |
| 50 | **136.9** | 180 ms | 260 ms | 320 ms | 53 ms | 0 |
| 100 | 125.9 | 500 ms | 660 ms | 770 ms | 160 ms | 0 |

Server-side p99, from Prometheus, evaluated at each level's end time:

| Users | `/route` handler | feature fetch | scoring |
|---|---|---|---|
| 10 | 70 ms | 49 ms | 24 ms |
| 25 | 177 ms* | 98 ms | 24 ms |
| 50 | 422 ms* | 249 ms | 43 ms |
| 100 | 984 ms* | **≥ 250 ms** (top bucket) | 89 ms |

\* *Higher than the client-side p99, which is impossible in reality.* These are
bucket-interpolation artifacts: `histogram_quantile` assumes observations are
spread evenly inside a bucket, and the route histogram jumps 100 → 250 → 500 →
1000 ms, so estimates inside those wide buckets overshoot. Use Locust for
end-to-end latency, and use the server-side numbers only to compare fetch
against scoring.

The fetch histogram's highest finite bucket is 250 ms, so at u100 it can only
report "at least 250 ms". Adding `0.5, 1.0` to `FEATURE_FETCH_LATENCY`'s buckets
fixes this for later runs without changing any reading below 250 ms.

### 9.4 Reading the throughput curve

Locust is a **closed loop**: each simulated user waits for a response before
sending its next request. One user's cycle is

```
/route latency + /outcome latency + think time (between(0.05, 0.2), mean 125 ms)
```

so N users can produce at most N ÷ cycle time. At u10 that predicts
10 ÷ (34 + 8 + 125 ms) ≈ 60 req/s, and the measured value was 59.6. At low load
the users' own pacing limits throughput, not the service.

The curve has three regions:

- **10 → 25 users:** throughput more than doubles (59.6 → 128.4). The service
  still has headroom.
- **25 → 50 users:** throughput gains only 7% while p50 rises 3.5×. This is the
  knee. **The ceiling is about 137 `/route` req/s** (about 274 req/s including
  `/outcome`).
- **50 → 100 users:** throughput **falls** 8% while p99 more than doubles. This
  is *retrograde scaling*: extra concurrency does not just queue, it makes the
  system less efficient. Some shared resource is being fought over, and the
  fight costs more than it delivers.

Little's Law (1.3) puts the in-flight count at saturation at roughly
149 req/s × 0.155 s ≈ 23 concurrent `/route` requests across three replicas.
`/outcome`, which only increments a few counters, rose from 7 ms to 160 ms
median. It does almost no work, so that time is queueing behind `/route`.

### 9.5 Resource utilization at saturation

`docker stats` snapshot during a 50-user run:

| Container | CPU (100% = one core) | Threads (PIDS) |
|---|---|---|
| app1 / app2 / app3 | **131–136% each** | 290–338 each |
| redis | 39% | 6 |
| nginx | 5% | 17 |
| prometheus / grafana | < 1% | — |

In total that's about 4.4 of 16 cores. **The host is not saturated, and neither
is Redis.** Redis is single-threaded, so it would be pinned near 100% if it were
the limit. It is idle more than half the time.

### 9.6 Diagnosis

**What Part 8 got right:** feature fetch dominates. Fetch p99 grows 49 → 98 →
249 → ≥250 ms with load, while scoring stays flat at 24 ms until saturation.

**What Part 8 got wrong:** the mechanism. Part 8 attributed fetch time to Redis
round-trip latency queueing inside Redis. The data rules that out: Redis is at
39% CPU. The time is spent on the app side of each round-trip.

**The most consistent explanation is GIL contention inside each replica.**

- `/route` and `/outcome` are plain `def` handlers, not `async def`, so FastAPI
  runs them in a thread pool. Many requests run concurrently **inside one Python
  process**.
- Python's GIL lets only one thread execute Python code at a time. Each replica
  is therefore capped at about one core of Python work. The ~130% CPU is that
  one core plus work done outside the GIL (C extensions, socket I/O).
- Each of the ~80 Redis calls in a fetch releases the GIL to wait on the
  socket, gets its reply in about 0.1 ms, and then **waits to re-acquire the
  GIL** behind every other runnable request thread. That handoff happens about
  80 times per request, and each wait grows with the number of threads. So
  fetch time scales with concurrency even though Redis itself is fast.
- More concurrency means more threads, more handoffs and more context
  switching. That accounts for the retrograde drop at u100.

At saturation, the 3 replicas served ~137 `/route` + ~137 `/outcome` req/s on
about 3 cores of Python. That works out to **roughly 20 ms of single-core Python
CPU per route+outcome cycle**, and that number is what any efficiency fix has
to reduce.

**Second finding: scoring is not sub-millisecond.** Part 8 predicted scoring
"likely sub-millisecond". It is 24 ms at p99 even at low load. A prime suspect
is **OpenMP thread oversubscription**. LightGBM's `predict` starts an OpenMP
team sized to the core count (16) **per calling thread**, so a thread pool of
request threads can create hundreds of OpenMP threads per process. That matches
the ~300 PIDS per app container: roughly 900 threads competing for 16 cores, to
score 4 rows each. Pandas overhead on tiny DataFrames (the `DataFrame` build,
column selection and the sklearn preprocessor) likely contributes too.
*Status: confirmed by experiment B2 (9.12).* At idle each app container has
**51** threads. Under 50-user load it has **290–338**, so load creates about 250
extra threads per process. FastAPI's thread pool (AnyIO's default limit is 40)
accounts for at most 40 of them. The remaining ~200 match OpenMP teams being
created by request threads. Setting `OMP_NUM_THREADS=1` is the experiment that
confirms or rules it out.

### 9.7 Traffic-meter clamping

`read_system_tps()` clamps whenever the last 60 seconds averaged under
`SYSTEM_TPS_MIN ÷ TRAFFIC_SCALE` ≈ 8,458 ÷ 800 ≈ **10 `/route` req/s**.

- **Steady state never clamps.** 60–137 req/s scales to about 48k–110k, inside
  the training range of 8,458–308,555.
- **No `high` clamps at any level,** so `TRAFFIC_SCALE=800` fits this load
  range.
- **`low` clamps: 1,331** over the 16 minutes ending at u100's end. These are
  cold starts after idle gaps (u10-rerun after a 2.7-minute gap, u25 after a
  2-minute gap). Each needs about 600 requests (10 req/s × 60 s) before the
  window average recovers, which is consistent with the total. Levels started
  within 17–34 s of the previous one still had a full window, so they should not
  have clamped.

That is about 1.5% of requests, all in the first seconds of a run, and outside
the `[2m]` windows used for the server-side numbers. It does not affect the
baseline.

### 9.8 The plan for the throughput loop

The bottleneck is **Python CPU per replica, made worse by contention**, not
Redis capacity. Candidate changes, roughly by effort:

| | Change | Why it should help |
|---|---|---|
| A | Remove the ~8 Redis writes per request that feed counters nothing reads | Fewer commands means less CPU and fewer GIL handoffs per request |
| B | `OMP_NUM_THREADS=1` for the app containers | Removes the suspected OpenMP oversubscription in scoring (tests 9.6's hypothesis) |
| C | Redis pipelining in `build_route_features()` | ~80 GIL handoffs become ~1 per request. Expected to be the largest single gain |
| D | More replicas | 12 of 16 cores and over half of Redis are idle. This adds capacity rather than efficiency, so it goes last, where it can't hide inefficiency |

**One change per experiment**, each measured with the identical procedure from
9.1, so every gain is attributable to one cause. Each experiment gets its own
commit and its own `loadtest/<name>_u<N>_stats.csv` files.

### 9.9 Re-baseline on the instrumented code (`baseline2`)

After the first baseline, the service gained observability: structured JSON
logging (sampled at `LOG_SAMPLE_RATE=0.01`), `/outcome` request counters,
calibration-monitoring metrics, and finer histogram buckets. That adds a little
Python work per request, and Python CPU is the bottleneck. So the experiments
are measured against a fresh baseline on that code, not against 9.3. Otherwise
the first experiment would silently include two changes.

Commit `c2bb639`, run with the Appendix A runbook (`RUN=baseline2`).

| Users | `/route` req/s | p50 | p95 | p99 | Failures | Baseline 1 (req/s / p50 / p99) |
|---|---|---|---|---|---|---|
| 10 | 61.4 | 28 ms | 54 ms | 72 ms | 1 (502) | 59.6 / 31 / 76 |
| 25 | 130.9 | 48 ms | 89 ms | 110 ms | 0 | 128.4 / 51 / 130 |
| 50 | **138.9** | 180 ms | 250 ms | 280 ms | 0 | 136.9 / 180 / 320 |
| 100 | 126.9 | 490 ms | 630 ms | 710 ms | 0 | 125.9 / 500 / 770 |

Server-side p99 at each level's end time:

| Users | `/route` handler | feature fetch | scoring |
|---|---|---|---|
| 10 | 67 ms | 49 ms | 15 ms |
| 25 | 100 ms | 96 ms | 18 ms |
| 50 | 295 ms | 270 ms | 43 ms |
| 100 | 742 ms | 886 ms* | 87 ms |

\* Fetch cannot exceed the route handler it is part of. The fetch histogram
still has a wide 500–1000 ms bucket, so read this one as "roughly 500–750 ms".

**The instrumentation cost nothing measurable, and this sets the noise
floor.** The curve has the same shape and the same ~138 req/s ceiling, and
`baseline2` came out 1–2% *faster* despite doing slightly more work. Together
with the 3% spread between the two warm-ups (9.2), run-to-run noise on this
laptop is about **±3%**. An experiment has to beat that before its gain counts
as real.

**The finer buckets corrected one number from 9.3.** Scoring p99 at low load
is about **15 ms**, not 24 ms. The earlier value was an interpolation inside the
coarse 10–25 ms bucket. The code was the same; the measurement got sharper. The
finer route buckets also brought server-side route p99 in line with Locust
(67 vs 72 ms, 295 vs 280 ms), removing the overshoot noted in 9.3. At
saturation, fetch is about **90%** of the route p99 (270 of 295 ms at u50).

**Per-level resource use** (`docker stats` snapshot during each level):

| Users | App CPU (each) | App threads (each) | Redis CPU |
|---|---|---|---|
| idle | ~0% | 51 | 4% |
| 10 | 52–59% | 114–130 | 19% |
| 25 | 112–120% | 178–194 | 34% |
| 50 | 130–132% | 306–322 | 40% |
| 100 | 132% | **578** | 46% |

- **App CPU plateaus at ~131% from u50, exactly where throughput plateaus.** This
  is the per-process ceiling from 9.6: about one GIL-bound core plus work done
  outside the GIL.
- **Threads keep climbing after CPU stops, in steps of ~16 = the core count.**
  (578 − 51) ÷ 16 ≈ 33 request threads at u100, and (306 − 51) ÷ 16 ≈ 16 at
  u50. FastAPI's thread pool is capped at 40, so it cannot produce 578 threads
  on its own. One 16-thread OpenMP team per request thread that has called
  `predict` fits the numbers. Across 3 replicas at u100, that's ~1,700 threads on
  16 cores. (Experiment B tested this with `OMP_NUM_THREADS=1` and found that
  the environment variable doesn't reach the predict path. See 9.11. Experiment
  B2 confirmed it: see 9.12.)
- **Redis peaks at 46%.** It has not been the bottleneck at any load level.

**Clamping:** `increase(...[13m])` over the sweep is **0**. With the sweep
started straight after the warm-up, the traffic meter never left the training
range. (The cold-start clamps after `FLUSHALL` fall in the warm-up, outside the
window.)

**The one failure: a 502 at u10.** The app logged no `internal_error`, so the
request never failed inside FastAPI. A 502 is nginx reporting that the upstream
connection broke. The likely cause is a **keepalive race**: nginx keeps idle
upstream connections open (`keepalive 128`, and nginx's default
`keepalive_timeout` is 60 s), while uvicorn closes idle connections after **5 s**.
If nginx sends a request on a connection at the moment uvicorn closes it, the
request fails, and nginx does not retry it because POST is not idempotent. Low
load (u10) leaves connections idle longest, which fits where it happened. The
rate is 1 in ~90,000 requests, so it doesn't affect the measurements. In a
payments service, though, a dropped routing request is a real defect. The fix
(uvicorn's `--timeout-keep-alive` longer than nginx's upstream
`keepalive_timeout`, or the reverse) changes connection behaviour, so it is
**deferred until after the throughput experiments**, to keep them single-variable.
*Status: confirmed.* nginx's error log shows exactly one
`recv() failed (104: Connection reset by peer) while reading response header from upstream`
for a `POST /route`, which is the signature of the upstream closing a reused
keepalive connection.

### 9.10 Experiment A — remove unread request counters (`expA_deadwrites`)

**Change:** `record_routing_decision()` wrote four per-minute `<scope>:req:<minute>`
counters (an `INCR` plus an `EXPIRE` each, so 8 Redis commands per `/route`) that
nothing reads anymore. Rolling rates moved to `resolved` in Part 1, and
utilization moved to the traffic meter. The writes and the now-unused
`record_request()` were deleted. `record_bank_amount()` stays, because
`get_bank_avg_amount()` reads it.

**Verification before measuring:** `grep` showed no reader of `"req"` in
`serving/`, and the parity test was unchanged: 22 pass, 4 known gaps, 0 fail
(of 26 features).

**Prediction:** about 120 Redis commands per route+outcome cycle, so 8 fewer is
about 7%. Expected a 3–7% throughput gain.

| Users | req/s | p50 | p99 | baseline2 (req/s / p50 / p99) | Δ req/s |
|---|---|---|---|---|---|
| 10 | 61.4 | 27 ms | 66 ms | 61.4 / 28 / 72 | 0% (think-time bound) |
| 25 | 132.1 | 46 ms | 110 ms | 130.9 / 48 / 110 | +0.9% |
| 50 | **141.1** | 170 ms | 270 ms | 138.9 / 180 / 280 | +1.6% |
| 100 | 129.0 | 470 ms | 670 ms | 126.9 / 490 / 710 | +1.7% |

| | baseline2 u50 / u100 | expA u50 / u100 |
|---|---|---|
| Redis CPU | 40% / 46% | 39% / 42% |
| App CPU (each) | ~131% / ~132% | ~130% / ~130% |
| App threads (each) | 306–322 / 578 | 274–306 / 562–578 |
| fetch p99 | 270 / 886 ms | 272 / 895 ms |
| scoring p99 | 43 / 87 ms | 43 / 88 ms |

**Result: inside the noise.** Throughput rose 1.5–2%, below the ±3% threshold
(9.9), so no gain can be claimed. The direction was consistent at every level
(throughput slightly up, p50 and p99 slightly down), and Redis CPU per request
fell by about 5–10%, in line with removing ~7% of its commands. So the change
probably has a small real effect that this setup can't resolve.

Fetch and scoring p99 were unchanged, as expected. The removed writes ran after
scoring, outside both timers, so this confirms the change touched nothing else.
App CPU stayed pinned at ~130% and the thread pattern from 9.9 was unchanged.

**What it teaches.** Redis load dropped about as much as the command count
predicted, but throughput barely moved. So the per-request Python cost is **not
dominated by issuing Redis commands**. A large share sits elsewhere in the
request: building the pandas DataFrame, the sklearn preprocessor, pydantic
validation and response building. That tempers the expectation for experiment
C. Pipelining removes ~80 round-trips and their GIL handoffs, but it cannot
remove Python work that isn't part of a round-trip.

The change is kept regardless: it is correct, it reduces Redis load, and parity
shows the model's inputs are unchanged.

### 9.11 Experiment B — `OMP_NUM_THREADS=1` (`expB_omp`)

**Change:** `OMP_NUM_THREADS=1` added to the image's `ENV` block in
`serving/Dockerfile`, to cap OpenMP at one thread per process.

**Prediction:** if the load-driven threads in 9.9 were OpenMP teams, thread
counts under load should stay near the idle count plus the thread pool (≤ 40),
instead of reaching ~578, and scoring should get cheaper.

| Users | req/s | p50 | p99 | expA (req/s / p50 / p99) |
|---|---|---|---|---|
| 10 | 61.7 | 27 ms | 63 ms | 61.4 / 27 / 66 |
| 25 | 132.4 | 46 ms | 110 ms | 132.1 / 46 / 110 |
| 50 | 140.0 | 170 ms | 280 ms | 141.1 / 170 / 270 |
| 100 | 129.3 | 470 ms | 670 ms | 129.0 / 470 / 670 |

Scoring p99 was unchanged (15 / 18 / 43 / 88 ms), as were fetch p99, app CPU
(~130% at saturation) and Redis CPU.

| Users | App threads, expA | App threads, expB | Load-created (minus idle), expA → expB |
|---|---|---|---|
| idle | 51 | **6** | |
| 10 | 114 | 53–69 | 63 → 47–63 |
| 25 | 162–178 | 117 | 111–127 → 111 |
| 50 | 274–306 | 261–277 | 223–255 → 255–271 |
| 100 | 562–578 | 517–533 | 511–527 → 511–527 |

**Result: no performance change, and the prediction failed in an informative
way.** The environment variable removed 45 *idle* threads: the thread pools that
OpenMP and numpy's OpenBLAS create at import. The threads created *by load*
were identical, about 527 at u100 in both runs, still growing by about 16 per
concurrent request.

**Why:** the model was trained with `LGBMClassifier(n_jobs=-1)`
(`ml/lightgbm_tuned.py:706`), and that parameter is pickled with the estimator.
On every `predict_proba`, LightGBM's sklearn wrapper resolves `n_jobs=-1` to
the CPU count (16) and passes it to the booster as an **explicit**
`num_threads`. An explicit thread count overrides `OMP_NUM_THREADS`, which only
sets OpenMP's *default*. So experiment B never reached the code path it was
meant to change. The OpenMP-teams hypothesis from 9.6 is therefore **untested,
not refuted**. The ×16-per-request-thread scaling still fits it exactly.

The lesson: a training-time parallelism setting silently became a serving-time
setting. `n_jobs=-1` is right for training on a big batch and wrong for scoring
4 rows from dozens of concurrent request threads.

`OMP_NUM_THREADS=1` stays in the image. It is harmless, removes 45 idle threads
per replica, and keeps numpy's thread pool from oversubscribing once scoring is
single-threaded. The actual fix is experiment B2.

### 9.12 Experiment B2 — single-threaded scoring (`expB2_njobs`)

**Change:** `_model.set_params(n_jobs=1)` right after the model is loaded in
`serving/main.py`, so every `predict_proba` passes `num_threads=1`. Predictions
are unchanged: LightGBM parallelizes over rows, and each row's score is computed
identically regardless of thread count.

| Users | req/s | p50 | p99 | expB (req/s / p50 / p99) | Δ req/s vs expB | Δ req/s vs baseline2 |
|---|---|---|---|---|---|---|
| 10 | 61.6 | 27 ms | 62 ms | 61.7 / 27 / 63 | 0% (think-time bound) | 0% |
| 25 | 134.8 | 44 ms | 99 ms | 132.4 / 46 / 110 | +1.8% | +3.0% |
| 50 | **145.4** | 160 ms | 260 ms | 140.0 / 170 / 280 | **+3.9%** | **+4.7%** |
| 100 | 134.8 | 450 ms | 630 ms | 129.3 / 470 / 670 | **+4.3%** | **+6.2%** |

| Users | scoring p99, expB → B2 | fetch p99 (B2) | App threads, expB → B2 | App CPU (B2) |
|---|---|---|---|---|
| idle | | | 6 → 6 | ~4% |
| 10 | 14.7 → 14.8 ms | 48 ms | 53–69 → **8** | 41–48% |
| 25 | 17.6 → 15.6 ms | 96 ms | 117 → **12–13** | 100–114% |
| 50 | 43.4 → **19.9 ms** | 272 ms | 261–277 → **20–21** | ~128% |
| 100 | 87.5 → **43.9 ms** | 834 ms* | 517–533 → **37–45** | 128–134% |

\* The wide 500–1000 ms fetch bucket again; read it as "roughly 500–700 ms".

**Result: the first gain above the noise floor, and the thread hypothesis is
confirmed.**

- **Threads:** the load-driven thread explosion is gone. At u100, each app
  peaks at 37–45 threads, which is 6 idle plus up to 40 request threads (AnyIO's
  thread-pool limit). The 9.9 hypothesis is **confirmed**: each request thread
  that called `predict` had its own 16-thread OpenMP team.
- **Throughput:** +3.9% at u50 and +4.3% at u100 versus expB, both above the
  ±3% threshold, and consistent at every saturated level. The ceiling is now
  ~145 `/route` req/s (+4.7% over `baseline2`).
- **Scoring under load halved** (43 → 20 ms at u50, 88 → 44 ms at u100). That is
  the coordination and oversubscription cost that has now been removed.
- **Scoring at low load did not move (~15 ms).** So the ~15 ms floor is not
  OpenMP. It is the rest of the scoring block: selecting DataFrame columns, the
  sklearn preprocessor's `transform`, and LightGBM's input conversion, all
  Python/pandas overhead on a 4-row frame. That is a separate future target.

**What did not change:**

- **App CPU still caps at ~130%.** The GIL ceiling is untouched. The gain came
  from wasting less of that one core on thread coordination, not from using
  more cores.
- **Retrograde scaling remains:** 145.4 → 134.8 req/s from u50 to u100 (−7%,
  against −8% before). So it was never caused by OpenMP. It is contention among
  the request threads themselves.
- **Fetch is now ~94% of route p99 at saturation** (272 of 289 ms at u50). It is
  the dominant cost by far, which makes pipelining (C) the next target.

---

## Part 10 — What Is Still Not Done

Stated plainly so nothing in the repo overstates itself:

1. **Only the baselines have been measured.** Part 9 records them. The README
   section still has a placeholder for the summary.
2. **Rare 502s from an nginx/uvicorn keepalive race** (9.9). Fix deferred until
   after the throughput experiments.
3. **Redis pipelining** — now backed by measurement (9.6), and step C of the
   throughput plan (9.8).
4. **CacheX swap-in** — separate repo, not started. The feature-fetch vs.
   scoring latency split exists specifically so this comparison is measurable
   when it happens.
5. **Closed-loop offline evaluation** — the offline routing simulation still
   assumes one routing decision doesn't affect future route health. The
   *deployed* service is a genuine closed loop, so load testing measures this
   directly and the offline version matters less than it did.
6. **Cost-aware ranking** — `/route` maximizes success probability, not expected
   profit. This matches what Razorpay's paper describes for their production
   system, and their paper lists cost-aware routing as future work too.
7. **Timezone anchoring** — `hour_of_day` is derived from UTC at serving time,
   while the simulator's traffic curve is in unanchored "local" time. Harmless
   on synthetic data; a real deployment would need the serving clock aligned to
   whatever timezone the traffic pattern was learned in.
8. **No authentication, rate limiting, or TLS.** Appropriate for a portfolio
   project; worth naming as a known gap rather than pretending otherwise.

---

## Part 11 — Explaining This Out Loud

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

**"What did the load test show?"**
Three replicas on a 16-core laptop topped out at about 137 routing decisions
per second, with throughput actually falling at higher concurrency. I'd
predicted Redis round-trips would be the bottleneck, but `docker stats` showed
Redis at 39% CPU while each app replica sat at about 130%. Each replica is one
Python process running its handlers in a thread pool, so the ~80 Redis calls
per request each have to re-acquire the GIL behind every other request thread.
The cost was contention in the app, not Redis. Scoring was also 24 ms instead of
the sub-millisecond I expected, and about 300 threads per container point to
LightGBM's OpenMP oversubscribing the cores.

**"What would you do next?"**
Work through the measured bottleneck one change at a time, re-running the
identical sweep after each: remove dead Redis writes, pin OpenMP to one thread,
then pipeline the Redis reads so ~80 GIL handoffs become one. Only then add
replicas. After that, swap Redis for the C++ cache I'm building and re-run the
identical load test to measure the difference.

---

## Appendix A — Load-Test Runbook

The procedure from 9.1 as copy-paste commands. Every run, baseline or
experiment, uses exactly these steps, so results stay comparable. Run
everything from the project root.

### A.1 Name the run

One name per run. It prefixes every output file:

```bash
RUN=baseline2        # later: expA_deadwrites, expB_omp, expC_pipeline, ...
```

`RUN` only exists in the terminal where you set it. If you open a new terminal,
set it again.

### A.2 Pre-flight (every time)

```bash
cd /media/sreyan/D/01_Dev/Projects/FluxRoute
source .venv-load/bin/activate
```

```bash
git status                          # code under test must be committed
git rev-parse --short HEAD          # record this hash with the results
```

```bash
docker compose up -d --build        # no-op if nothing changed; rebuilds if serving/ did
docker compose ps                   # 7 containers Up; redis + app1-3 (healthy)
```

```bash
curl -s -w '\n' -X POST localhost:8080/route -H 'Content-Type: application/json' \
  -d '{"amount":1200,"bank":"HDFC","network":"Visa","payment_method":"CARD","merchant":"Shopping","device":"Android","risk":"LOW"}'
```

The smoke test must return a 200 with a `chosen_route_id` before any load is
applied (see 9.2, the discarded run).

### A.3 Reset and warm up (not recorded)

```bash
docker compose exec redis redis-cli FLUSHALL
```

```bash
locust -f loadtest/locustfile.py --host http://localhost:8080 --headless \
  -u 50 -r 10 -t 2m --only-summary
```

Start the sweep **immediately** after the warm-up. A gap of more than 60 s
empties the traffic meter's window and causes cold-start clamping (9.7).

### A.4 The sweep (about 13 minutes)

```bash
rm -f loadtest/${RUN}_endtimes.txt
for u in 10 25 50 100; do
  locust -f loadtest/locustfile.py --host http://localhost:8080 --headless \
    -u $u -r $(( u < 25 ? u : 25 )) -t 3m --csv loadtest/${RUN}_u$u --only-summary
  echo "$u $(date +%s)" >> loadtest/${RUN}_endtimes.txt
done
```

The four levels run back to back with no gaps. After each level ends, its user
count and Unix end time are appended to `loadtest/${RUN}_endtimes.txt`, which
A.5 uses to query Prometheus.

Optional, in a second terminal during the u50 level (minutes 3.5–6 of the
sweep):

```bash
docker stats --no-stream
```

### A.5 Collect the results

**Client side (Locust):** req/s, p50, p95, p99 and failures per level:

```bash
for u in 10 25 50 100; do
  awk -F, -v u=$u '$2=="POST /route" {printf "u%-4s req/s=%6.1f  p50=%4s  p95=%4s  p99=%4s  fails=%s\n", u, $10, $12, $17, $19, $4}' \
    loadtest/${RUN}_u${u}_stats.csv
done
```

**Server side (Prometheus):** p99 of route handler, fetch and scoring,
evaluated at each level's end time over its last 2 minutes. Values are in
seconds:

```bash
while read u t; do
  for m in route_latency_seconds feature_fetch_seconds scoring_latency_seconds; do
    printf 'u%-4s %-25s ' "$u" "$m"
    curl -s localhost:9090/api/v1/query \
      --data-urlencode "query=histogram_quantile(0.99, sum by (le) (rate(fluxroute_${m}_bucket[2m])))" \
      --data-urlencode "time=$t" | grep -o '"[^"]*"\]'
  done
done < loadtest/${RUN}_endtimes.txt
```

**Traffic-meter clamping** over the whole sweep (an empty result means no
clamping). The 13-minute window covers the ~12-minute sweep plus the end of the
warm-up, but not the warm-up's cold start, which always clamps right after
`FLUSHALL`:

```bash
curl -s localhost:9090/api/v1/query \
  --data-urlencode 'query=sum by (direction) (increase(fluxroute_system_tps_clamped_total[13m]))' \
  --data-urlencode "time=$(tail -1 loadtest/${RUN}_endtimes.txt | cut -d' ' -f2)"
```

Prometheus keeps 15 days of data by default, so these A.5 queries can be re-run
later, as long as the Prometheus container hasn't been removed.
`docker compose down` removes it and its data.

### A.6 Record and commit

Add the results to the report, then commit only the summary files:

```bash
git add loadtest/${RUN}_u*_stats.csv loadtest/${RUN}_endtimes.txt deployment-report.md
git commit -m "Load test: ${RUN}"
```

Locust's `_stats_history.csv`, `_failures.csv` and `_exceptions.csv` files are
git-ignored. The history file is large, and the other two are empty when
the run is clean (and if they aren't, the run isn't valid anyway).

### A.7 Experiment discipline

- **One change per experiment.** Each experiment gets its own commit, so every
  measured difference has exactly one cause.
- **Compare against the run immediately before it,** not only against the
  original baseline. Improvements stack, and each step's gain should be
  measured on its own.
- **Same host conditions:** close heavy applications (browsers with many tabs,
  IDE indexing, other containers) before a run. They all share the same CPU.
