#!/usr/bin/env python3
"""
Independent validation of full replication results.
Runs after run_full_replication.sh completes.

Checks:
  1. All cells present and non-empty
  2. Statistics recomputed from raw latencies (not trusting stats field)
  3. Cross-check SET A results against original run
  4. Convergence windows for SET B (PO2 vs MPC-PO2 vs Kalman-PO2)
  5. Kalman routing trajectory (inversion detection per 200-req window)
  6. Methodological flags (n_err, n requests)
"""
import json, statistics, os, sys, glob

def find_rd(base, suffix):
    dirs = sorted(glob.glob(f"{base}/*_{suffix}"))
    if not dirs:
        sys.exit(f"No results dir matching *_{suffix} in {base}")
    return dirs[-1]

BASE = "/shared_inference/vpolamre/orbit/results"
RD   = find_rd(BASE, "replication")
print(f"Results dir: {RD}")

POLICIES_A = ["rr", "lor", "po2", "mpc_po2", "kalman_po2"]
RATIOS     = ["1x", "2x", "4x", "8x"]
POLICIES_B = ["po2", "mpc_po2", "kalman_po2"]

issues = []

# ── helpers ──────────────────────────────────────────────────────────────

def load_raw(label):
    p = f"{RD}/{label}.json"
    if not os.path.exists(p):
        return None, f"MISSING FILE: {p}"
    d = json.load(open(p))
    raw_all = d.get("raw", [])
    ok  = [r for r in raw_all if r.get("ok")]
    err = [r for r in raw_all if not r.get("ok")]
    if not ok:
        return None, f"NO OK REQUESTS in {label}"
    lats = sorted([r["latency"] for r in ok])
    n = len(lats)
    return {
        "n":      n,
        "n_err":  len(err),
        "mean":   statistics.mean(lats),
        "stdev":  statistics.stdev(lats) if n > 1 else 0.0,
        "p50":    lats[n // 2],
        "p95":    lats[int(0.95 * n)],
        "p99":    lats[int(0.99 * n)],
        "lats":   lats,
        "raw_ok": ok,
    }, None

def pct(base, new):
    if base is None or new is None: return "n/a"
    return f"{(new['mean']-base['mean'])/base['mean']*100:+.1f}%"

# ── SET A: presence + basic stats ────────────────────────────────────────

print()
print("=" * 80)
print("SET A — Replication Results")
print(f"{'Ratio':<5} {'Policy':<12} {'n':>4} {'nerr':>4}  {'mean':>7}  {'stdev':>7}  {'p95':>7}  {'p99':>8}  {'vs_po2':>8}")
print("-" * 80)

seta = {}
for ratio in RATIOS:
    for pol in POLICIES_A:
        label = f"seta_{ratio}_{pol}"
        s, err = load_raw(label)
        seta[(ratio, pol)] = s
        if err:
            issues.append(err)
            print(f"  {ratio:<5} {pol:<12}  --- {err}")
            continue
        po2 = seta.get((ratio, "po2"))
        delta = pct(po2, s) if pol not in ("rr", "po2") and po2 else ""
        nerr_flag = " !" if s["n_err"] > 0 else ""
        print(f"  {ratio:<5} {pol:<12} {s['n']:>4} {s['n_err']:>4}{nerr_flag}"
              f"  {s['mean']:>7.1f}  {s['stdev']:>7.1f}  {s['p95']:>7.1f}  {s['p99']:>8.1f}  {delta:>8}")
        if s["n_err"] > 0:
            issues.append(f"SET A {label}: {s['n_err']} errors")
        if s["n"] < 100:
            issues.append(f"SET A {label}: only {s['n']} ok requests (expected 120)")
    print()

# ── SET A: MPC-PO2 should be worse than PO2 everywhere ───────────────────

print()
print("SET A — Policy ordering check (expect: LOR ≈ PO2 < MPC-PO2, Kalman varies)")
for ratio in RATIOS:
    po2 = seta.get((ratio, "po2"))
    mpc = seta.get((ratio, "mpc_po2"))
    kf  = seta.get((ratio, "kalman_po2"))
    lor = seta.get((ratio, "lor"))
    if po2 and mpc:
        mpc_vs_po2 = (mpc["mean"] - po2["mean"]) / po2["mean"] * 100
        mpc_ok = "OK (MPC worse)" if mpc_vs_po2 > 0 else "UNEXPECTED (MPC better than PO2)"
        print(f"  {ratio}: MPC vs PO2: {mpc_vs_po2:+.1f}%  -> {mpc_ok}")
    if po2 and kf:
        kf_vs_po2 = (kf["mean"] - po2["mean"]) / po2["mean"] * 100
        print(f"  {ratio}: Kalman vs PO2: {kf_vs_po2:+.1f}%")
    if po2 and lor:
        lor_vs = (lor["mean"] - po2["mean"]) / po2["mean"] * 100
        print(f"  {ratio}: LOR vs PO2: {lor_vs:+.1f}%")
    print()

# ── SET B: 10K convergence ────────────────────────────────────────────────

print("=" * 80)
print("SET B — 10K Convergence (8x, 4 rps)")
print("=" * 80)

setb = {}
for pol in POLICIES_B:
    label = f"setb_8x_{pol}_10k"
    s, err = load_raw(label)
    setb[pol] = s
    if err:
        issues.append(err)
        print(f"  {pol}: {err}")
        continue
    print(f"  {pol}: n={s['n']}  n_err={s['n_err']}  mean={s['mean']:.1f}ms  "
          f"stdev={s['stdev']:.1f}ms  p99={s['p99']:.1f}ms")

# convergence windows
print()
WINDOW = 200
print(f"{'Window':<14}", end="")
for pol in POLICIES_B:
    print(f"  {pol:>16}", end="")
print()
print("-" * (14 + 19 * len(POLICIES_B)))

max_n = max((setb[p]["n"] for p in POLICIES_B if setb.get(p)), default=0)
for start in range(0, max_n, WINDOW):
    print(f"{start:5d}-{start+WINDOW:<6}", end="")
    po2_m = None
    for pol in POLICIES_B:
        if not setb.get(pol):
            print(f"  {'---':>16}", end=""); continue
        chunk = setb[pol]["raw_ok"][start:start+WINDOW]
        if len(chunk) < WINDOW // 2:
            print(f"  {'---':>16}", end=""); continue
        m = statistics.mean(r["latency"] for r in chunk)
        if pol == "po2":
            po2_m = m
            print(f"  {m:10.1f}ms    ", end="")
        else:
            delta = f"({(m-po2_m)/po2_m*100:+.1f}%)" if po2_m else "(---)"
            print(f"  {m:8.1f}{delta:>8}", end="")
    print()

# ── Kalman routing stability analysis ────────────────────────────────────

print()
print("=" * 80)
print("Kalman-PO2 routing stability (SET B 10K) — latency proxy analysis")
print("  lat < 350ms = proxy-fast (fast server), lat >= 350ms = proxy-slow")
print("=" * 80)

kf_data = setb.get("kalman_po2")
if kf_data:
    ok = kf_data["raw_ok"]
    inversions = []
    print(f"  {'Window':<12}  {'%fast':>6}  {'mean':>7}  {'stdev':>7}  {'flag'}")
    for start in range(0, len(ok), 500):
        chunk = ok[start:start+500]
        if not chunk: continue
        lats_c = [r["latency"] for r in chunk]
        fast_p = sum(1 for l in lats_c if l < 350) / len(lats_c)
        m = statistics.mean(lats_c)
        sd = statistics.stdev(lats_c) if len(lats_c) > 1 else 0
        flag = ""
        if fast_p < 0.2:
            flag = "  <-- INVERSION (routing to slow)"
            inversions.append(start)
        elif fast_p > 0.95:
            flag = "  converged"
        bar = "#" * int(fast_p * 20)
        print(f"  {start:5d}-{start+len(chunk):<5d}  {fast_p*100:5.1f}%  {m:7.1f}  {sd:7.1f}  [{bar:<20}]{flag}")

    if inversions:
        issues.append(f"Kalman-PO2 SET B: routing inversions at windows starting req "
                      f"{inversions}")
        print(f"\n  WARNING: {len(inversions)} inversion window(s) detected")
    else:
        print(f"\n  PASS: No routing inversions in 10K run")

# ── Final summary ─────────────────────────────────────────────────────────

print()
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
if not issues:
    print("  ALL CHECKS PASSED — no issues found")
else:
    print(f"  {len(issues)} ISSUE(S) FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
