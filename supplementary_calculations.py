"""Supplementary calculations for:
"Robust Modular Floating Wetland: Design & Wave-Tank Evaluation"

Computes, from the raw down-sampled (0.2 Hz) tracking time series:
  - Wave-tank performance metrics (mean radial drift, 95th-percentile radial
    drift, RMS heading deviation, cross-channel tension coefficient of
    variation) with Moving Block Bootstrap (MBB) 95% confidence intervals.
  - Convergence of cumulative mean radial drift at 5/10/20/30 min.
  - An indicative bill-of-materials cost summary for the reference 50-mat
    installation.

Only the tension-guide and distributed-modular configurations were
instrumented in the wave tank (perimeter-rigid overturned in preliminary
trials and was evaluated only through field deployment), so those are the
two runs analyzed here.

Inputs:  TensionGuide.csv, DistributedModular.csv (this directory)
Outputs: results.json (consumed by plot.py) + a console summary.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from html_report import generate_html_report

DATA_DIR = Path(__file__).resolve().parent

# Keys match the test_id prefixes used in the raw data / plot.py's results.json schema.
CONFIGS = {
    "B_TensionGuide": DATA_DIR / "TensionGuide.csv",
    "C_DistributedModular": DATA_DIR / "DistributedModular.csv",
}
DISPLAY_NAME = {
    "B_TensionGuide": "Tension-guide",
    "C_DistributedModular": "Distributed-modular",
}
TENSION_COLS = ["tension_N1", "tension_N2", "tension_N3"]

DT_SEC = 5.0            # sampling interval of the primary (down-sampled) analysis
N_BOOT = 5000           # bootstrap replicates
ACF_THRESHOLD = 0.05    # block-length stopping criterion
MAX_LAG = 200           # search ceiling for the ACF-derived block length
RNG_SEED = 12345


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 60.0
    return df


# --------------------------------------------------------------------------
# Moving Block Bootstrap
# --------------------------------------------------------------------------
def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[: n - lag], x[lag:]) / denom
    return acf


def acf_block_length(drift_cm: np.ndarray, threshold: float = ACF_THRESHOLD,
                      max_lag: int = MAX_LAG) -> tuple[int, bool]:
    """Smallest lag at which |ACF(drift)| first falls below `threshold`.

    Returns (block_length, crossed). If the ACF never drops below the
    threshold within `max_lag`, the search ceiling is returned as a lower
    bound (crossed=False)."""
    acf = autocorrelation(drift_cm, max_lag)
    below = np.where(np.abs(acf[1:]) < threshold)[0]
    if below.size == 0:
        return max_lag, False
    return int(below[0] + 1), True


def block_indices(n: int, block_length: int, n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]


def tension_cov(mean_tensions: np.ndarray) -> float:
    """Cross-channel coefficient of variation: std (ddof=1) / mean of the
    per-channel mean tensions."""
    mean_tensions = np.asarray(mean_tensions, dtype=float)
    return mean_tensions.std(ddof=1) / mean_tensions.mean()


def point_estimates(df: pd.DataFrame) -> dict:
    mean_tensions = df[TENSION_COLS].mean().to_numpy()
    return {
        "n_samples": len(df),
        "mean_drift_cm": float(df["r_cm"].mean()),
        "p95_drift_cm": float(np.percentile(df["r_cm"], 95)),
        "rms_heading_deg": float(np.sqrt(np.mean(df["heading_deg"] ** 2))),
        "mean_tensions_N": mean_tensions.tolist(),
        "tension_cov": float(tension_cov(mean_tensions)),
    }


def bootstrap_ci(df: pd.DataFrame, block_length: int, n_boot: int = N_BOOT,
                  rng: np.random.Generator | None = None) -> dict:
    rng = rng if rng is not None else np.random.default_rng(RNG_SEED)
    n = len(df)
    n_blocks = int(np.ceil(n / block_length))
    r = df["r_cm"].to_numpy()
    h = df["heading_deg"].to_numpy()
    tensions = df[TENSION_COLS].to_numpy()

    mean_drift = np.empty(n_boot)
    p95_drift = np.empty(n_boot)
    rms_heading = np.empty(n_boot)
    cov_t = np.empty(n_boot)

    for b in range(n_boot):
        idx = block_indices(n, block_length, n_blocks, rng)
        mean_drift[b] = r[idx].mean()
        p95_drift[b] = np.percentile(r[idx], 95)
        rms_heading[b] = np.sqrt(np.mean(h[idx] ** 2))
        cov_t[b] = tension_cov(tensions[idx].mean(axis=0))

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {
        "mean_drift_ci": ci(mean_drift),
        "p95_drift_ci": ci(p95_drift),
        "rms_heading_ci": ci(rms_heading),
        "tension_cov_ci": ci(cov_t),
        "tension_cov_mbb_mean": float(cov_t.mean()),
    }


def convergence_table(df: pd.DataFrame, checkpoints_min=(5, 10, 20, 30)) -> dict:
    return {t: float(df.loc[df["elapsed_min"] <= t, "r_cm"].mean()) for t in checkpoints_min}


def time_above_threshold(df: pd.DataFrame, drift_thresh_cm: float | None = None,
                          tension_thresh_N: float | None = None) -> float | None:
    """Fraction of samples where radial drift or any single tether tension
    exceeds an operational limit. Supply `drift_thresh_cm`/`tension_thresh_N`
    (the platform's design limits) to compute it; otherwise returns None."""
    if drift_thresh_cm is None and tension_thresh_N is None:
        return None
    exceed = np.zeros(len(df), dtype=bool)
    if drift_thresh_cm is not None:
        exceed |= df["r_cm"].to_numpy() > drift_thresh_cm
    if tension_thresh_N is not None:
        exceed |= (df[TENSION_COLS].to_numpy() > tension_thresh_N).any(axis=1)
    return float(exceed.mean())


def analyze_config(key: str, path: Path, rng: np.random.Generator) -> dict:
    df = load_run(path)
    pe = point_estimates(df)
    block_len, crossed = acf_block_length(df["r_cm"].to_numpy())
    ci = bootstrap_ci(df, block_len, rng=rng)
    return {
        "config": DISPLAY_NAME[key],
        "n": pe["n_samples"],
        "dt_sec": DT_SEC,
        "block_lag_samples": block_len,
        "block_lag_sec": block_len * DT_SEC,
        "block_lag_is_lower_bound": not crossed,
        "mean_drift_cm": pe["mean_drift_cm"],
        "mean_drift_ci": ci["mean_drift_ci"],
        "p95_drift_cm": pe["p95_drift_cm"],
        "p95_drift_ci": ci["p95_drift_ci"],
        "rms_heading_deg": pe["rms_heading_deg"],
        "rms_heading_ci": ci["rms_heading_ci"],
        "mean_tensions_N": pe["mean_tensions_N"],
        "tension_cov": pe["tension_cov"],
        "tension_cov_mbb_mean": ci["tension_cov_mbb_mean"],
        "tension_cov_ci": ci["tension_cov_ci"],
        "convergence_mean_drift_cm": convergence_table(df),
    }


# --------------------------------------------------------------------------
# Indicative cost summary
# --------------------------------------------------------------------------
def indicative_cost_summary(n_mats: int = 50, plants_per_mat: int = 65,
                             mat_cost_low: float = 240.0, mat_cost_high: float = 280.0,
                             plant_cost_low: float = 3.0, plant_cost_high: float = 4.0) -> dict:
    """Rough bill-of-materials for the reference installation. `mat_cost_*`
    already bundles the HDPE mat, concrete anchors, rope, hardware, nursery
    containers, plant stock and installation labor (~15 min/mat) per mat;
    `plant_cost_*` is the per-plant unit price folded into that total and is
    reported for reference only -- it is NOT additive with the per-mat cost."""
    return {
        "n_mats": n_mats,
        "plants_per_mat": plants_per_mat,
        "total_plants": n_mats * plants_per_mat,
        "cost_per_mat_usd": [mat_cost_low, mat_cost_high],
        "cost_per_plant_usd_reference_only": [plant_cost_low, plant_cost_high],
        "total_cost_usd": [n_mats * mat_cost_low, n_mats * mat_cost_high],
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_report(results: dict) -> None:
    cfgs = results["per_config"]

    print("\n" + "=" * 66)
    print("Wave-tank performance metrics, 30-min runs")
    print("=" * 66)
    rows = [
        ("Mean radial drift (cm)", "mean_drift_cm", "mean_drift_ci"),
        ("95th-pct drift (cm)", "p95_drift_cm", "p95_drift_ci"),
        ("RMS heading (deg)", "rms_heading_deg", "rms_heading_ci"),
    ]
    for label, key, ci_key in rows:
        print(f"\n{label}")
        for cfg in cfgs.values():
            v, (lo, hi) = cfg[key], cfg[ci_key]
            print(f"  {cfg['config']:22s} {v:6.3f}  (95% CI: {lo:.3f}, {hi:.3f})")

    print("\nCross-channel tension CoV")
    for cfg in cfgs.values():
        print(f"  {cfg['config']:22s} {cfg['tension_cov']:.2e}")

    print("\nMean line tensions (N)")
    for cfg in cfgs.values():
        vals = ", ".join(f"{v:.3f}" for v in cfg["mean_tensions_N"])
        print(f"  {cfg['config']:22s} {vals}")

    print("\nACF-derived block length L*")
    for cfg in cfgs.values():
        bound = ">=" if cfg["block_lag_is_lower_bound"] else "="
        print(f"  {cfg['config']:22s} {bound} {cfg['block_lag_sec']:.0f} s "
              f"({cfg['block_lag_samples']} samples)")

    print("\n" + "=" * 66)
    print("Convergence of mean radial drift (cm)")
    print("=" * 66)
    checkpoints = sorted(next(iter(cfgs.values()))["convergence_mean_drift_cm"])
    header = f"{'Elapsed (min)':<15s}" + "".join(f"{cfg['config']:<24s}" for cfg in cfgs.values())
    print(header)
    for t in checkpoints:
        line = f"{t:<15d}"
        for cfg in cfgs.values():
            line += f"{cfg['convergence_mean_drift_cm'][t]:<24.2f}"
        print(line)

    cs = results["cost_summary"]
    print("\n" + "=" * 66)
    print("Indicative cost summary")
    print("=" * 66)
    print(f"  {cs['n_mats']} mats x {cs['plants_per_mat']} plants/mat = {cs['total_plants']} plants")
    print(f"  Cost per mat:   ${cs['cost_per_mat_usd'][0]:,.0f} - ${cs['cost_per_mat_usd'][1]:,.0f}")
    print(f"  Total cost:     ${cs['total_cost_usd'][0]:,.0f} - ${cs['total_cost_usd'][1]:,.0f}")


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    results = {"per_config": {}}
    for key, path in CONFIGS.items():
        if not path.exists():
            print(f"  [skip] {DISPLAY_NAME[key]}: {path.name} not found")
            continue
        print(f"Analyzing {DISPLAY_NAME[key]} ({path.name}) ...")
        results["per_config"][key] = analyze_config(key, path, rng)

    results["cost_summary"] = indicative_cost_summary()

    out_path = DATA_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    html_path = DATA_DIR / "report.html"
    generate_html_report(results, html_path)

    print_report(results)
    print(f"\nWrote {out_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
