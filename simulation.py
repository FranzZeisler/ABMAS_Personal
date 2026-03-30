"""
simulation.py – Four-phase experimental pipeline for the ABMAS airport simulation.

Phase 1 : Determine minimum replications N via CV convergence.
Phase 2 : Run BASELINE and TUG scenarios for N replications each.
Phase 3 : Statistical hypothesis testing (normality + significance + effect size).
Phase 4 : One-At-a-Time (OAT) sensitivity analysis and Tornado diagram.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from scipy import stats

matplotlib.use("Agg")

from run_me import run_airport_simulation

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
for _d in (RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared base configuration (locked in for Phases 2-4)
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    "mode": "TUG",           # overridden per phase
    "spawn_interval": 5.0,   # busy traffic
    "number_of_tugs": 5,     # fixed fleet size
    "mean_delay": 3.0,       # standard mean delay
    "variance_delay": 2.0,   # standard variance
    "tug_speed": 1.0,
    "tug_comm_dist": 2.0,
    "simulation_time": 500,
    "visualization": False,
}

# Significance level for hypothesis tests
ALPHA = 0.05

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silent_run(config):
    """Run a single simulation, suppressing all stdout output."""
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        return run_airport_simulation(config)


def _mean_taxi_time(results):
    """Extract the system-average Total Taxi Time from a run result dict."""
    aircraft = results.get("aircraft", [])
    if not aircraft:
        return float("nan")
    return sum(a["Total Taxi Time (s)"] for a in aircraft) / len(aircraft)


def _kpi_series(results, kpi_col):
    """Return a list of per-aircraft KPI values from a run result dict."""
    return [a[kpi_col] for a in results.get("aircraft", []) if kpi_col in a]


# ===========================================================================
# Phase 1 – Simulation Stability and Convergence
# ===========================================================================

def phase1_convergence(convergence_window=10, cv_threshold=0.01):
    """
    Run the worst-case scenario iteratively until the Coefficient of Variation
    of the system-average Total Taxi Time stabilises.

    The worst-case scenario uses the highest variability parameters:
    - Minimum spawn interval (peak traffic)
    - Maximum variance in stochastic delays

    Convergence criterion: CV fluctuates by less than *cv_threshold* (1 %)
    across *convergence_window* (10) consecutive runs.

    Returns
    -------
    N : int
        The minimum number of replications required for stable results.
    cv_history : list of float
        The CV value recorded after each run.
    """
    print("\n" + "=" * 60)
    print("PHASE 1 – Simulation Stability and Convergence Analysis")
    print("=" * 60)

    # Worst-case scenario: peak traffic, highest delay variance
    worst_case = {
        **BASE_CONFIG,
        "mode": "TUG",
        "spawn_interval": 5.0,    # most traffic
        "variance_delay": 2.0,    # highest variance
    }

    kpi_values = []      # cumulative list of mean Total Taxi Time per run
    cv_history = []      # CV after each run

    run_idx = 0
    converged = False

    while not converged:
        run_idx += 1
        cfg = {**worst_case, "seed": run_idx}
        results = _silent_run(cfg)
        mean_tt = _mean_taxi_time(results)
        if not math.isnan(mean_tt):
            kpi_values.append(mean_tt)

        n = len(kpi_values)
        if n >= 2:
            mu = np.mean(kpi_values)
            sigma = np.std(kpi_values, ddof=1)
            cv = sigma / mu if mu != 0 else float("inf")
        else:
            cv = float("inf")

        cv_history.append(cv)
        print(f"  Run {run_idx:3d} | mean TTT = {mean_tt:.2f} s | "
              f"cumulative CV = {cv:.6f}")

        # Check convergence: last `convergence_window` CV values all within threshold
        if n >= convergence_window + 1:
            recent = cv_history[-convergence_window:]
            max_fluctuation = max(recent) - min(recent)
            if max_fluctuation < cv_threshold:
                converged = True

        # Safety cap to prevent infinite loops
        if run_idx >= 500:
            print("  [WARNING] Reached 500 runs without convergence; using N=500.")
            break

    N = run_idx
    print(f"\n  --> Convergence reached after N = {N} replications.")
    print(f"      Final CV = {cv_history[-1]:.6f}")

    # Save convergence trace
    cv_df = pd.DataFrame({"Run": range(1, len(cv_history) + 1), "CV": cv_history})
    cv_df.to_csv(os.path.join(RESULTS_DIR, "phase1_cv_convergence.csv"), index=False)

    # Plot convergence curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cv_df["Run"], cv_df["CV"], color="#2196F3", linewidth=1.5, label="CV")
    ax.axhline(cv_threshold, color="#F44336", linestyle="--", linewidth=1.2,
               label=f"Threshold = {cv_threshold:.0%}")
    ax.axvline(N, color="#4CAF50", linestyle=":", linewidth=1.5,
               label=f"N = {N}")
    ax.set_title("Phase 1 – CV Convergence of Mean Total Taxi Time",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Runs")
    ax.set_ylabel("Coefficient of Variation (σ/μ)")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.savefig(os.path.join(FIGURES_DIR, "phase1_cv_convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Convergence figure saved.")

    return N, cv_history


# ===========================================================================
# Phase 2 – Base Scenario Execution
# ===========================================================================

def phase2_run_base_scenarios(N):
    """
    Execute BASELINE and TUG simulations for N replications using matching seeds.

    Returns
    -------
    baseline_results : list of dict  (one dict per replication)
    tug_results      : list of dict  (one dict per replication)
    Each dict contains 'seed', 'mean_taxi_time', 'mean_engine_on_time',
    and the raw 'aircraft' list from run_airport_simulation.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 2 – Base Scenario Execution  (N = {N} replications)")
    print("=" * 60)

    seeds = list(range(1, N + 1))

    baseline_results = []
    tug_results = []

    for i, seed in enumerate(seeds, 1):
        print(f"  Replication {i}/{N} (seed={seed}) ...", end="")

        # --- BASELINE run ---
        base_cfg = {**BASE_CONFIG, "mode": "BASELINE", "number_of_tugs": 0, "seed": seed}
        base_res = _silent_run(base_cfg)
        base_aircraft = base_res.get("aircraft", [])
        baseline_results.append({
            "seed": seed,
            "mean_taxi_time": np.mean([a["Total Taxi Time (s)"] for a in base_aircraft]) if base_aircraft else float("nan"),
            "mean_engine_on_time": np.mean([a["Engine-On Time (s)"] for a in base_aircraft]) if base_aircraft else float("nan"),
            "aircraft": base_aircraft,
        })

        # --- TUG run (same seed) ---
        tug_cfg = {**BASE_CONFIG, "mode": "TUG", "seed": seed}
        tug_res = _silent_run(tug_cfg)
        tug_aircraft = tug_res.get("aircraft", [])
        tug_results.append({
            "seed": seed,
            "mean_taxi_time": np.mean([a["Total Taxi Time (s)"] for a in tug_aircraft]) if tug_aircraft else float("nan"),
            "mean_engine_on_time": np.mean([a["Engine-On Time (s)"] for a in tug_aircraft]) if tug_aircraft else float("nan"),
            "aircraft": tug_aircraft,
        })

        print(f"  BASELINE TTT={baseline_results[-1]['mean_taxi_time']:.1f}s  "
              f"TUG TTT={tug_results[-1]['mean_taxi_time']:.1f}s")

    # Persist per-replication summary
    summary_rows = []
    for b, tg in zip(baseline_results, tug_results):
        summary_rows.append({
            "Seed": b["seed"],
            "BASELINE_Mean_Taxi_Time": b["mean_taxi_time"],
            "BASELINE_Mean_Engine_On": b["mean_engine_on_time"],
            "TUG_Mean_Taxi_Time": tg["mean_taxi_time"],
            "TUG_Mean_Engine_On": tg["mean_engine_on_time"],
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS_DIR, "phase2_base_scenario_results.csv"), index=False
    )
    print(f"\n  Results saved to {RESULTS_DIR}/phase2_base_scenario_results.csv")

    return baseline_results, tug_results


# ===========================================================================
# Phase 3 – Statistical Hypothesis Testing
# ===========================================================================

def _normality_test(data, label):
    """Apply Shapiro-Wilk (N<50) or Kolmogorov-Smirnov (N>=50) test."""
    n = len(data)
    if n < 50:
        stat, p = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        # Two-sided KS test against a fitted normal distribution
        mu, sigma = np.mean(data), np.std(data, ddof=1)
        stat, p = stats.kstest(data, "norm", args=(mu, sigma))
        test_name = "Kolmogorov-Smirnov"
    is_normal = p >= ALPHA
    print(f"    {label}: {test_name} stat={stat:.4f}, p={p:.4f} "
          f"-> {'NORMAL' if is_normal else 'NON-NORMAL'}")
    return is_normal, test_name, stat, p


def _cohens_d(x1, x2):
    """Calculate Cohen's d using pooled standard deviation."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return (np.mean(x2) - np.mean(x1)) / pooled_std


def phase3_hypothesis_testing(baseline_results, tug_results):
    """
    Evaluate two hypotheses:
      H1 – Electric tugs reduce Total Taxi Time.
      H2 – Electric tugs reduce Engine-On Time.

    Steps:
      1. Normality check per KPI per mode.
      2. Welch's t-test (both normal) or Mann-Whitney U (non-normal).
      3. Reject H0 if p < ALPHA.
      4. Calculate Cohen's d effect size.

    Returns
    -------
    report : dict  keyed by KPI name, containing test results.
    """
    print("\n" + "=" * 60)
    print("PHASE 3 – Statistical Hypothesis Testing")
    print(f"  Significance level α = {ALPHA}")
    print("=" * 60)

    # Extract per-replication mean KPIs (one value per run)
    base_ttt = [r["mean_taxi_time"] for r in baseline_results if not math.isnan(r["mean_taxi_time"])]
    tug_ttt = [r["mean_taxi_time"] for r in tug_results if not math.isnan(r["mean_taxi_time"])]
    base_eot = [r["mean_engine_on_time"] for r in baseline_results if not math.isnan(r["mean_engine_on_time"])]
    tug_eot = [r["mean_engine_on_time"] for r in tug_results if not math.isnan(r["mean_engine_on_time"])]

    report = {}
    for kpi_name, base_data, tug_data in [
        ("Total Taxi Time (s)", base_ttt, tug_ttt),
        ("Engine-On Time (s)", base_eot, tug_eot),
    ]:
        print(f"\n  --- {kpi_name} ---")
        print(f"    BASELINE: mean={np.mean(base_data):.2f}  std={np.std(base_data, ddof=1):.2f}  n={len(base_data)}")
        print(f"    TUG:      mean={np.mean(tug_data):.2f}  std={np.std(tug_data, ddof=1):.2f}  n={len(tug_data)}")

        print("  Normality tests:")
        base_normal, base_test, base_stat, base_p = _normality_test(base_data, "  BASELINE")
        tug_normal, tug_test, tug_stat, tug_p = _normality_test(tug_data, "  TUG     ")

        both_normal = base_normal and tug_normal
        if both_normal:
            # Welch's unpaired t-test (does not assume equal variances)
            test_stat, p_value = stats.ttest_ind(base_data, tug_data, equal_var=False)
            chosen_test = "Welch's t-test"
        else:
            test_stat, p_value = stats.mannwhitneyu(base_data, tug_data, alternative="two-sided")
            chosen_test = "Mann-Whitney U"

        reject_h0 = p_value < ALPHA
        d = _cohens_d(base_data, tug_data)

        print(f"  Significance test: {chosen_test}")
        print(f"    stat = {test_stat:.4f},  p = {p_value:.6f}")
        print(f"    {'REJECT H0' if reject_h0 else 'FAIL TO REJECT H0'} (α={ALPHA})")
        print(f"    Cohen's d = {d:.4f}")

        report[kpi_name] = {
            "baseline_mean": np.mean(base_data),
            "tug_mean": np.mean(tug_data),
            "base_normality_test": base_test,
            "base_normality_stat": base_stat,
            "base_normality_p": base_p,
            "base_is_normal": base_normal,
            "tug_normality_test": tug_test,
            "tug_normality_stat": tug_stat,
            "tug_normality_p": tug_p,
            "tug_is_normal": tug_normal,
            "significance_test": chosen_test,
            "test_stat": test_stat,
            "p_value": p_value,
            "reject_h0": reject_h0,
            "cohens_d": d,
        }

    # Save report
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "KPI"})
    report_df.to_csv(os.path.join(RESULTS_DIR, "phase3_hypothesis_report.csv"), index=False)

    # Plot KPI distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (kpi_name, base_data, tug_data) in zip(
        axes,
        [
            ("Total Taxi Time (s)", base_ttt, tug_ttt),
            ("Engine-On Time (s)", base_eot, tug_eot),
        ],
    ):
        ax.hist(base_data, bins="auto", alpha=0.6, color="#FF9800", label="BASELINE", edgecolor="white")
        ax.hist(tug_data, bins="auto", alpha=0.6, color="#2196F3", label="TUG", edgecolor="white")
        rec = report[kpi_name]
        ax.set_title(f"{kpi_name}\n"
                     f"p={rec['p_value']:.4f}  d={rec['cohens_d']:.3f}",
                     fontsize=11)
        ax.set_xlabel(kpi_name)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.3)

    fig.suptitle("Phase 3 – KPI Distributions: BASELINE vs TUG", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase3_kpi_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Distribution figure saved.")
    print(f"  Report saved to {RESULTS_DIR}/phase3_hypothesis_report.csv")

    return report


# ===========================================================================
# Phase 4 – Local Sensitivity Analysis (OAT)
# ===========================================================================

def phase4_sensitivity_analysis(N):
    """
    One-At-a-Time (OAT) sensitivity analysis.

    For each parameter, compute the mean Total Taxi Time at:
      - base value
      - base * (1 + 0.05)
      - base * (1 - 0.05)

    Sensitivity index S = (ΔX / X̄) / (ΔP / P)

    Parameters perturbed:
      - number_of_tugs  (tug fleet size)
      - tug_speed
      - variance_delay  (battery-discharge / delay variance proxy)
      - tug_comm_dist   (conflict detection radius)

    Returns
    -------
    sensitivity_df : DataFrame with columns [Parameter, S_plus, S_minus, S_mean]
    """
    print("\n" + "=" * 60)
    print(f"PHASE 4 – OAT Sensitivity Analysis  (N = {N} replications)")
    print("=" * 60)

    PERTURB = 0.05  # 5 % perturbation
    KPI = "Total Taxi Time (s)"

    # Parameters to perturb: (label, config_key, base_value)
    parameters = [
        ("Tug Fleet Size",     "number_of_tugs", BASE_CONFIG["number_of_tugs"]),
        ("Tug Speed",          "tug_speed",       BASE_CONFIG["tug_speed"]),
        ("Delay Variance",     "variance_delay",  BASE_CONFIG["variance_delay"]),
        ("Tug Comm. Distance", "tug_comm_dist",   BASE_CONFIG["tug_comm_dist"]),
        ("Mean Delay",         "mean_delay",      BASE_CONFIG["mean_delay"]),
        ("Spawn Interval",     "spawn_interval",  BASE_CONFIG["spawn_interval"]),
    ]

    # --- Compute base mean KPI (N replications, TUG mode) ---
    print(f"\n  Computing base mean {KPI} ...")
    base_kpis = []
    for seed in range(1, N + 1):
        cfg = {**BASE_CONFIG, "mode": "TUG", "seed": seed}
        res = _silent_run(cfg)
        aircraft = res.get("aircraft", [])
        if aircraft:
            base_kpis.append(np.mean([a[KPI] for a in aircraft]))

    X_base = np.mean(base_kpis) if base_kpis else float("nan")
    print(f"    Base {KPI} = {X_base:.2f} s")

    rows = []
    for label, key, P_base in parameters:
        print(f"\n  Perturbing '{label}' (base={P_base}) ...")

        for direction, factor in [("+5%", 1 + PERTURB), ("-5%", 1 - PERTURB)]:
            P_new = P_base * factor
            # Keep integer parameters as integers (e.g. fleet size)
            if isinstance(P_base, int):
                P_new = max(1, round(P_new))

            kpis = []
            for seed in range(1, N + 1):
                cfg = {**BASE_CONFIG, "mode": "TUG", key: P_new, "seed": seed}
                res = _silent_run(cfg)
                aircraft = res.get("aircraft", [])
                if aircraft:
                    kpis.append(np.mean([a[KPI] for a in aircraft]))

            X_new = np.mean(kpis) if kpis else float("nan")
            delta_X = X_new - X_base
            delta_P_ratio = (P_new - P_base) / P_base if P_base != 0 else float("nan")
            S = (delta_X / X_base) / delta_P_ratio if (X_base != 0 and delta_P_ratio != 0) else float("nan")

            print(f"    {direction}: P={P_new:.3f}  X={X_new:.2f}  ΔX={delta_X:.2f}  S={S:.4f}")
            rows.append({
                "Parameter": label,
                "Direction": direction,
                "P_base": P_base,
                "P_new": P_new,
                "X_base": X_base,
                "X_new": X_new,
                "Delta_X": delta_X,
                "S": S,
            })

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df.to_csv(os.path.join(RESULTS_DIR, "phase4_sensitivity.csv"), index=False)

    # --- Tornado Diagram ---
    # For each parameter use max(|S+|, |S-|) as the bar width, keeping sign of larger magnitude
    tornado_rows = []
    for label, _, _ in parameters:
        sub = sensitivity_df[sensitivity_df["Parameter"] == label]
        if sub.empty:
            continue
        s_vals = sub["S"].values
        # Keep both values; the bar spans from min to max
        s_lo = float(np.nanmin(s_vals))
        s_hi = float(np.nanmax(s_vals))
        tornado_rows.append({"Parameter": label, "S_lo": s_lo, "S_hi": s_hi,
                              "S_abs_max": max(abs(s_lo), abs(s_hi))})

    tornado_df = pd.DataFrame(tornado_rows).sort_values("S_abs_max")

    fig, ax = plt.subplots(figsize=(10, max(4, len(tornado_df) * 0.7 + 1)))
    y_pos = range(len(tornado_df))
    colors_pos = "#2196F3"
    colors_neg = "#FF9800"

    for i, row in enumerate(tornado_df.itertuples()):
        lo, hi = row.S_lo, row.S_hi
        # Negative bar (perturbation reduced KPI or vice-versa)
        if lo < 0:
            ax.barh(i, abs(lo), left=lo, color=colors_neg, edgecolor="white", height=0.55)
        # Positive bar
        if hi > 0:
            ax.barh(i, hi, left=0, color=colors_pos, edgecolor="white", height=0.55)
        # If both same sign
        if lo >= 0 and hi >= 0:
            ax.barh(i, hi - lo, left=lo, color=colors_pos, edgecolor="white", height=0.55)
        if lo <= 0 and hi <= 0:
            ax.barh(i, abs(lo) - abs(hi), left=hi, color=colors_neg, edgecolor="white", height=0.55)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tornado_df["Parameter"].tolist())
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sensitivity Index  S = (ΔX/X̄) / (ΔP/P)")
    ax.set_title("Phase 4 – Tornado Diagram: OAT Sensitivity Analysis",
                 fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_pos, label="+5% perturbation"),
                       Patch(facecolor=colors_neg, label="-5% perturbation")]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase4_tornado_diagram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Sensitivity results saved to {RESULTS_DIR}/phase4_sensitivity.csv")
    print(f"  Tornado diagram saved to {FIGURES_DIR}/phase4_tornado_diagram.png")

    return sensitivity_df


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Phase 1 – Determine N
    # ------------------------------------------------------------------
    N, cv_history = phase1_convergence()

    # ------------------------------------------------------------------
    # Phase 2 – Run base scenarios
    # ------------------------------------------------------------------
    baseline_results, tug_results = phase2_run_base_scenarios(N)

    # ------------------------------------------------------------------
    # Phase 3 – Statistical testing
    # ------------------------------------------------------------------
    hypothesis_report = phase3_hypothesis_testing(baseline_results, tug_results)

    # ------------------------------------------------------------------
    # Phase 4 – Sensitivity analysis
    # ------------------------------------------------------------------
    sensitivity_df = phase4_sensitivity_analysis(N)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print(f"  N (replications) = {N}")
    print(f"  Results saved to : {RESULTS_DIR}/")
    print(f"  Figures saved to : {FIGURES_DIR}/")
    print("=" * 60)
