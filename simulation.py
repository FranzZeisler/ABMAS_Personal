import os
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count
from contextlib import redirect_stdout
from run_me import run_airport_simulation
from figures import generate_figures


def generate_scenarios():
    """Generates a full factorial design for all 5 factors."""

    # Check if results directory exists, if not create it
    if not os.path.exists("results"):
        os.makedirs("results")

    scenarios = []

    # Define the 5 factors
    spawn_intervals = [5.0, 10.0, 20.0]
    tug_fleet_sizes = [1, 2, 5, 10]
    mean_delays = [1.0, 3.0, 5.0]
    variance_delays = [0.5, 2.0]
    tug_comm_dists = [1.0, 2.0]

    # Defined fixed parameters
    simulation_time = 500  # seconds

    # Create all combinations of the 5 variable factors
    combinations = list(
        itertools.product(
            spawn_intervals,
            tug_fleet_sizes,
            mean_delays,
            variance_delays,
            tug_comm_dists,
        )
    )

    for interval, tug_count, delay, var_delay, comm_dist in combinations:
        param_id = f"Sp{interval}_Fl{tug_count}_Dl{delay}" f"_Vd{var_delay}_Cd{comm_dist}"

        # 1. Add the TUG version of this combination
        scenarios.append(
            {
                "scenario_id": f"TUG_{param_id}",
                "mode": "TUG",
                "spawn_interval": interval,
                "number_of_tugs": tug_count,
                "mean_delay": delay,
                "variance_delay": var_delay,
                "tug_comm_dist": comm_dist,
                "simulation_time": simulation_time,
                "visualization": False,
            }
        )

        # 2. Add the BASELINE version of this combination
        # (number_of_tugs, tug_comm_dist do not affect Baseline,
        # but we run it to observe how spawn/delay parameters affect throughput)
        scenarios.append(
            {
                "scenario_id": f"BASE_{param_id}",
                "mode": "BASELINE",
                "spawn_interval": interval,
                "number_of_tugs": 0,
                "mean_delay": delay,
                "variance_delay": var_delay,
                "tug_comm_dist": comm_dist,
                "simulation_time": simulation_time,
                "visualization": False,
            }
        )

    return scenarios


def worker(config):
    with open(os.devnull, "w") as f, redirect_stdout(f):
        try:
            results = run_airport_simulation(config)

            # --- NEW: Unpack the new dictionary structure from run_me.py ---
            ac_kpis = results.get("aircraft", [])
            df = pd.DataFrame(ac_kpis)

            if not df.empty:
                df["Scenario_ID"] = config["scenario_id"]
                df["Mode"] = config["mode"]
                df["Spawn_Interval"] = config["spawn_interval"]
                df["Fleet_Size"] = config["number_of_tugs"]
                df["Mean_Delay"] = config["mean_delay"]
                df["Variance_Delay"] = config["variance_delay"]
                df["Tug_Comm_Dist"] = config["tug_comm_dist"]

                # --- NEW: Add system and tug KPIs to the dataframe ---
                df["Throughput_per_Hour"] = results.get("throughput", 0.0)

                tugs_kpis = results.get("tugs", [])
                if tugs_kpis:
                    avg_towing = sum(t["Active Towing Time (s)"] for t in tugs_kpis) / len(tugs_kpis)
                    avg_deadhead = sum(t["Deadhead Time (s)"] for t in tugs_kpis) / len(tugs_kpis)
                    df["Avg_Tug_Towing_Time"] = avg_towing
                    df["Avg_Tug_Deadhead_Time"] = avg_deadhead
                else:
                    df["Avg_Tug_Towing_Time"] = 0.0
                    df["Avg_Tug_Deadhead_Time"] = 0.0

                return df
        except Exception as e:
            print(f"Simulation failed for scenario '{config.get('scenario_id', '?')}': {e}", flush=True)
            return pd.DataFrame()
    return pd.DataFrame()


if __name__ == "__main__":
    configs = generate_scenarios()
    print(f"Starting batch of {len(configs)} scenarios...")

    results_list = []
    with Pool(cpu_count()) as pool:
        for i, df in enumerate(pool.imap_unordered(worker, configs)):
            results_list.append(df)
            if i % 5 == 0:
                print(f"Progress: {i}/{len(configs)} simulations complete.")

    final_df = pd.concat(results_list, ignore_index=True)
    final_df.to_csv("results/full_factorial_results.csv", index=False)

    # --- NEW: Updated summary to aggregate the newly tracked KPIs ---
    summary = (
        final_df.groupby(["Mode", "Spawn_Interval", "Fleet_Size", "Mean_Delay", "Variance_Delay", "Tug_Comm_Dist"])
        .agg(
            {
                "Engine-On Time (s)": "mean",
                "Total Taxi Time (s)": "mean",
                "Wait Time for Tug (s)": "mean",
                "Throughput_per_Hour": "first",
                "Avg_Tug_Deadhead_Time": "first",
            }
        )
        .rename(columns={"Throughput_per_Hour": "Throughput (Ac/Hr)"})
    )

    print("\n--- Batch Results Summary ---")
    print(summary)
    # Reset index to save as CSV with columns instead of index labels
    summary.reset_index().to_csv("results/batch_summary.csv", index=False)

    # Generate output figures
    generate_figures(final_df)
