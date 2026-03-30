"""
figures.py – Visualisation of full-factorial simulation results.

Call ``generate_figures(df)`` after a batch run to produce a suite of
publication-quality PNG figures saved to the working directory.

The function expects a DataFrame containing at minimum the columns produced
by simulation.py's worker():
    Flight ID, Type, Total Taxi Time (s), Engine-On Time (s),
    Scenario_ID, Mode, Spawn_Interval, Fleet_Size, Mean_Delay,
    Variance_Delay, Tug_Comm_Dist, Wait Time for Tug (s),
    Throughput_per_Hour, Avg_Tug_Towing_Time, Avg_Tug_Deadhead_Time
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.use("Agg")  # non-interactive backend; works in batch / headless runs

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
MODE_PALETTE = {"TUG": "#2196F3", "BASELINE": "#FF9800"}
FACTOR_COLOUR = "#5C6BC0"

# Check if figures directory exists, if not create it
OUTPUT_DIR = "figures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


# ---------------------------------------------------------------------------
# Helper: aggregate per-scenario mean KPIs
# ---------------------------------------------------------------------------
def _scenario_means(df):
    """Return one row per Scenario_ID with mean KPIs and factor columns."""
    factor_cols = [
        "Scenario_ID",
        "Mode",
        "Spawn_Interval",
        "Fleet_Size",
        "Mean_Delay",
        "Variance_Delay",
        "Tug_Speed",
        "Tug_Comm_Dist",
    ]

    # Include new KPIs in aggregation if they exist
    kpi_cols = [
        "Engine-On Time (s)",
        "Total Taxi Time (s)",
        "Wait Time for Tug (s)",
        "Throughput_per_Hour",
        "Avg_Tug_Towing_Time",
        "Avg_Tug_Deadhead_Time",
    ]

    factor_cols = [c for c in factor_cols if c in df.columns]
    kpi_cols = [c for c in kpi_cols if c in df.columns]

    agg = df.groupby(factor_cols)[kpi_cols].mean().reset_index()
    # throughput = number of aircraft that completed (legacy count, keeping for safety)
    throughput = df.groupby(factor_cols)["Flight ID"].count().reset_index(name="Throughput_Count")
    return agg.merge(throughput, on=factor_cols)


# ---------------------------------------------------------------------------
# Figure 1 – Overall KPI comparison TUG vs BASELINE
# ---------------------------------------------------------------------------
def fig_overall_comparison(df):
    """Grouped bar chart: mean Engine-On Time and Total Taxi Time by mode."""
    means = df.groupby("Mode")[["Engine-On Time (s)", "Total Taxi Time (s)"]].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    means.T.plot(
        kind="bar",
        ax=ax,
        color=[MODE_PALETTE.get("BASELINE", "#FF9800"), MODE_PALETTE.get("TUG", "#2196F3")],
        edgecolor="white",
        width=0.55,
    )
    ax.set_title("Overall KPI Comparison: TUG vs BASELINE", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Time (s)")
    ax.set_xlabel("")
    ax.set_xticklabels(["Engine-On Time", "Total Taxi Time"], rotation=0)
    ax.legend(title="Mode")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, "fig01_overall_comparison.png")


# ----------------------------------------------------------------------------
# Figure 2 – Effect of Spawn Interval on KPIs
# ----------------------------------------------------------------------------
def fig_spawn_interval_effect(df):
    """Line plot showing KPIs across different Spawn Intervals."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, grp in df.groupby("Mode"):
        # We need scenario means here so we don't plot hundreds of dots vertically
        grp_mean = grp.groupby("Spawn_Interval")["Engine-On Time (s)"].mean()
        ax.plot(grp_mean.index, grp_mean.values, marker="o", label=mode, color=MODE_PALETTE.get(mode, "#000000"))

    ax.set_title("Effect of Traffic (Spawn Interval) on Engine-On Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Spawn Interval (s) [Lower = More Traffic]")
    ax.set_ylabel("Engine-On Time (s)")
    ax.legend(title="Mode")
    ax.grid(linestyle="--", alpha=0.4)
    _save(fig, "fig02_spawn_interval_effect.png")


# ---------------------------------------------------------------------------
# Figure 3 - Scatter KPIs per mode
# ---------------------------------------------------------------------------
def fig_scatter_kpis(df):
    """Scatter of individual aircraft Total Taxi Time vs Engine-On Time."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for mode, grp in df.groupby("Mode"):
        ax.scatter(
            grp["Total Taxi Time (s)"],
            grp["Engine-On Time (s)"],
            alpha=0.25,
            s=12,
            label=mode,
            color=MODE_PALETTE.get(mode, "#000000"),
        )
    ax.set_title("Engine-On Time vs Total Taxi Time (per aircraft)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Taxi Time (s)")
    ax.set_ylabel("Engine-On Time (s)")
    ax.legend(title="Mode")
    ax.grid(linestyle="--", alpha=0.4)
    _save(fig, "fig03_scatter_kpis.png")


# ---------------------------------------------------------------------------
# NEW: Figure 4 - Wait Time for Tug vs Fleet Size (Interaction Plot)
# ---------------------------------------------------------------------------
def fig_wait_time_vs_fleet(df):
    """Line chart showing how Fleet Size affects Wait Time across different Spawn Intervals."""
    tug_df = df[df["Mode"] == "TUG"]
    if tug_df.empty or "Wait Time for Tug (s)" not in tug_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Group by Fleet Size AND Spawn Interval to show the interaction
    agg = tug_df.groupby(["Fleet_Size", "Spawn_Interval"])["Wait Time for Tug (s)"].mean().unstack()

    agg.plot(kind="line", marker="o", ax=ax, linewidth=2)
    ax.set_title("Aircraft Wait Time vs. Tug Fleet Size", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fleet Size (Number of Tugs)")
    ax.set_ylabel("Mean Wait Time for Tug (s)")

    # Clean up the legend
    ax.legend(title="Spawn Interval (s)\n[Lower = Busiest]", loc="upper right")
    ax.grid(linestyle="--", alpha=0.4)

    # Ensure x-axis ticks match our discrete fleet sizes
    ax.set_xticks(tug_df["Fleet_Size"].unique())

    _save(fig, "fig04_wait_time_vs_fleet.png")


# ---------------------------------------------------------------------------
# NEW: Figure 5 - System Throughput Comparison
# ---------------------------------------------------------------------------
def fig_throughput_comparison(df):
    """Bar chart comparing aircraft throughput per hour."""
    if "Throughput_per_Hour" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Average throughput across combinations of mode and spawn interval
    agg = df.groupby(["Spawn_Interval", "Mode"])["Throughput_per_Hour"].mean().unstack()

    agg.plot(
        kind="bar",
        ax=ax,
        color=[MODE_PALETTE.get("BASELINE", "#FF9800"), MODE_PALETTE.get("TUG", "#2196F3")],
        edgecolor="white",
    )
    ax.set_title("System Throughput: TUG vs BASELINE", fontsize=13, fontweight="bold")
    ax.set_xlabel("Spawn Interval (s)")
    ax.set_ylabel("Throughput (Aircraft / Hour)")
    ax.legend(title="Mode")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=0)

    _save(fig, "fig05_throughput_comparison.png")


# ---------------------------------------------------------------------------
# NEW: Figure 6 - Tug Utilization (Active vs Deadhead)
# ---------------------------------------------------------------------------
def fig_tug_utilization(df):
    """Stacked bar chart showing Tug Time breakdown."""
    tug_df = df[df["Mode"] == "TUG"]
    if tug_df.empty or "Avg_Tug_Towing_Time" not in tug_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Group by Fleet Size
    agg = tug_df.groupby("Fleet_Size")[["Avg_Tug_Towing_Time", "Avg_Tug_Deadhead_Time"]].mean()

    agg.plot(kind="bar", stacked=True, ax=ax, color=["#4CAF50", "#F44336"], edgecolor="white", width=0.5)
    ax.set_title("Tug Time Breakdown by Fleet Size", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fleet Size")
    ax.set_ylabel("Mean Time per Tug (s)")
    ax.legend(["Active Towing Time", "Deadhead (Empty) Time"])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=0)

    _save(fig, "fig06_tug_utilization.png")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_figures(df):
    """
    Generate and save all output figures from a full-factorial results DataFrame.
    """
    if df is None or df.empty:
        print("No data available to generate figures.")
        return

    print("\nGenerating output figures...")
    fig_overall_comparison(df)
    fig_spawn_interval_effect(df)
    fig_scatter_kpis(df)

    # New Figures
    fig_wait_time_vs_fleet(df)
    fig_throughput_comparison(df)
    fig_tug_utilization(df)

    print("All figures saved to the /figures/ directory.")
