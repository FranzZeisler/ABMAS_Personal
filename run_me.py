"""
Run-me.py is the main file of the simulation. Run this file to run the simulation.
"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as timer
import random
import pygame as pg
from single_agent_planner import calc_heuristics
from visualization import map_initialization, map_running
from Aircraft import Aircraft
from tug import Tug
from independent import run_independent_planner
from prioritized import run_prioritized_planner
from cbs import run_CBS
from environment import AirportResourceManager
from fleet_manager import FleetManager


# %% SET SIMULATION PARAMETERS
# Input file names (used in import_layout) -> Do not change those unless you want to specify a new layout.
nodes_file = "nodes.xlsx"  # xlsx file with for each node: id, x_pos, y_pos, type
# xlsx file with for each edge: from  (node), to (node), length
edges_file = "edges.xlsx"

# Parameters that can be changed:
simulation_time = 500
# choose which planner to use (currently only Independent is implemented)
# planner = "Independent"
planner = "Prioritized"

# Visualization (can also be changed)
plot_graph = False  # show graph representation in NetworkX
visualization = True  # pygame visualization
visualization_speed = 0.1  # set at 0.1 as default

# Communication distance
tug_comm_dist = 2.0

# %%Function definitions


def import_layout(nodes_file, edges_file):
    """
    Imports layout information from xlsx files and converts this into dictionaries.
    INPUT:
        - nodes_file = xlsx file with node input data
        - edges_file = xlsx file with edge input data
    RETURNS:
        - nodes_dict = dictionary with nodes and node properties
        - edges_dict = dictionary with edges annd edge properties
        - start_and_goal_locations = dictionary with node ids for arrival runways, departure runways and gates
    """
    gates_xy = []  # lst with (x,y) positions of gates
    # lst with (x,y) positions of entry points of departure runways
    rwy_dep_xy = []
    # lst with (x,y) positions of exit points of arrival runways
    rwy_arr_xy = []

    script_dir = os.path.dirname(os.path.abspath(__file__))

    df_nodes = pd.read_excel(os.path.join(script_dir, nodes_file))
    df_edges = pd.read_excel(os.path.join(script_dir, edges_file))

    # Create nodes_dict from df_nodes
    nodes_dict = {}
    for i, row in df_nodes.iterrows():
        node_properties = {
            "id": row["id"],
            "x_pos": row["x_pos"],
            "y_pos": row["y_pos"],
            "xy_pos": (row["x_pos"], row["y_pos"]),
            "type": row["type"],
            "neighbors": set(),
        }
        node_id = row["id"]
        nodes_dict[node_id] = node_properties

        # Add node type
        if row["type"] == "rwy_d":
            rwy_dep_xy.append((row["x_pos"], row["y_pos"]))
        elif row["type"] == "rwy_a":
            rwy_arr_xy.append((row["x_pos"], row["y_pos"]))
        elif row["type"] == "gate":
            gates_xy.append((row["x_pos"], row["y_pos"]))

    # Specify node ids of gates, departure runways and arrival runways in a dict
    start_and_goal_locations = {"gates": gates_xy, "dep_rwy": rwy_dep_xy, "arr_rwy": rwy_arr_xy}

    # Create edges_dict from df_edges
    edges_dict = {}
    for i, row in df_edges.iterrows():
        edge_id = (row["from"], row["to"])
        from_node = edge_id[0]
        to_node = edge_id[1]
        start_end_pos = (nodes_dict[from_node]["xy_pos"], nodes_dict[to_node]["xy_pos"])
        edge_properties = {
            "id": edge_id,
            "from": row["from"],
            "to": row["to"],
            "length": row["length"],
            "weight": row["length"],
            "start_end_pos": start_end_pos,
        }
        edges_dict[edge_id] = edge_properties

    # Add neighbor nodes to nodes_dict based on edges between nodes
    for edge in edges_dict:
        from_node = edge[0]
        to_node = edge[1]
        nodes_dict[from_node]["neighbors"].add(to_node)

    return nodes_dict, edges_dict, start_and_goal_locations


def create_graph(nodes_dict, edges_dict, plot_graph=True):
    """
    Creates networkX graph based on nodes and edges and plots
    INPUT:
        - nodes_dict = dictionary with nodes and node properties
        - edges_dict = dictionary with edges annd edge properties
        - plot_graph = boolean (True/False) If True, function plots NetworkX graph. True by default.
    RETURNS:
        - graph = networkX graph object
    """
    graph = nx.DiGraph()  # create directed graph in NetworkX

    # Add nodes and edges to networkX graph
    for node in nodes_dict.keys():
        graph.add_node(
            node, node_id=nodes_dict[node]["id"], xy_pos=nodes_dict[node]["xy_pos"], node_type=nodes_dict[node]["type"]
        )

    for edge in edges_dict.keys():
        graph.add_edge(
            edge[0],
            edge[1],
            edge_id=edge,
            from_node=edges_dict[edge]["from"],
            to_node=edges_dict[edge]["to"],
            weight=edges_dict[edge]["length"],
        )

    # Plot networkX graph
    if plot_graph:
        plt.figure()
        node_locations = nx.get_node_attributes(graph, "xy_pos")
        nx.draw(graph, node_locations, with_labels=True, node_size=100, font_size=10)

    return graph


def run_airport_simulation(config):
    """
    Runs the airport simulation with the given configuration and returns KPI data.

    This function encapsulates the full simulation loop so it can be called
    programmatically (e.g., from simulation.py for batch runs) or interactively
    from __main__.

    INPUT:
        config : dict with the following keys:
            - mode              : "TUG" or "BASELINE"
            - spawn_interval    : seconds between scheduled aircraft (float)
            - number_of_tugs    : number of electric tugs to deploy (int, 0 for BASELINE)
            - mean_delay        : mean stochastic delay per aircraft in seconds (float)
            - variance_delay    : variance of stochastic delay per aircraft in seconds (float, optional, default 1.0)
            - tug_speed         : tug movement speed (float)
            - tug_comm_dist     : communication distance for tug coordination in map units (float, optional, default 2.0)
            - simulation_time   : total simulation duration in seconds (int/float)
            - visualization     : whether to show the pygame window (bool)
            - visualization_speed : pygame frame delay in seconds (float, optional)
    RETURNS:
        Dictionary containing 'aircraft' kpis, 'tugs' kpis, and overall 'throughput'.
    """
    # =============================================================================
    # 0. Unpack configuration
    # =============================================================================
    sim_mode = config.get("mode", "TUG")
    spawn_interval = config.get("spawn_interval", 10.0)
    number_of_tugs = config.get("number_of_tugs", 5)
    mean_delay = config.get("mean_delay", 2.0)
    variance_delay = config.get("variance_delay", 1.0)
    tug_speed_val = config.get("tug_speed", 1.0)
    tug_comm_dist_val = config.get("tug_comm_dist", 2.0)
    sim_time = config.get("simulation_time", 1000)
    show_vis = config.get("visualization", False)
    vis_speed = config.get("visualization_speed", 0.1)

    # =============================================================================
    # 0. Initialization
    # =============================================================================
    nodes_dict, edges_dict, start_and_goal_locations = import_layout(nodes_file, edges_file)

    # Initialize Resource Manager
    airport_manager = AirportResourceManager(nodes_dict)

    graph = create_graph(nodes_dict, edges_dict, plot_graph)
    heuristics = calc_heuristics(graph, nodes_dict)

    # Initialise the centralised Fleet Manager (uses pre-computed heuristics)
    fleet_manager = FleetManager(heuristics, nodes_dict)

    aircraft_lst = []  # List which can contain aircraft agents
    completed_kpis = []  # List to store data of finished aircraft

    print("Simulation Started")
    print(f"Total Gates: {len(airport_manager.gates)}")

    # Initialize tugs based on mode and configuration
    if sim_mode == "TUG":
        gate_ids = list(airport_manager.gates.keys())
        tug_lst = [
            Tug(101 + i, gate_ids[i % len(gate_ids)], nodes_dict, speed=tug_speed_val) for i in range(number_of_tugs)
        ]
        print(f"Mode: ELECTRIC TUGS ({number_of_tugs} tugs will dispatch and tow aircraft).")
    else:
        tug_lst = []  # No tugs in baseline mode
        print("Mode: BASELINE (No tugs, aircraft taxi independently).")

    unspawned_aircraft = []  # Waiting room for delayed aircraft
    last_scheduled_time = -spawn_interval  # First aircraft scheduled at t=0
    next_ac_id = 1

    # Open the visualization window if requested
    if show_vis:
        map_properties = map_initialization(nodes_dict, edges_dict)  # visualization properties

    # =============================================================================
    # 1. While loop and visualization
    # =============================================================================

    # Start of while loop
    running = True
    escape_pressed = False
    dt = 0.1  # should be factor of 0.5 (0.5/dt should be integer)
    t = 0

    while running:
        t = round(t, 2)

        # Check conditions for termination
        if t >= sim_time or escape_pressed:
            running = False
            if show_vis:
                pg.quit()
            print("Simulation Stopped")
            break

        # Visualization: Update map if visualization is true
        if show_vis:
            current_states = {}

            # 1. Determine which aircraft are currently being actively towed
            towed_ac_ids = [
                tug.assigned_ac.id for tug in tug_lst if tug.status == "Towing" and tug.assigned_ac is not None
            ]

            for ac in aircraft_lst:
                # Only send the aircraft to the visualizer if it IS NOT being towed right now
                if ac.status in ["Taxiing", "Coupled", "AtGate"] and ac.id not in towed_ac_ids:
                    current_states["AC_" + str(ac.id)] = {
                        "id": ac.id,
                        "ac_id": ac.id,
                        "xy_pos": ac.position,
                        "heading": ac.heading,
                        "type": "aircraft",
                    }

            for tug in tug_lst:
                if tug.status in ["Idle", "Dispatching", "Waiting", "Towing", "Returning"]:
                    current_states["TUG_" + str(tug.id)] = {
                        "id": tug.id,
                        "xy_pos": tug.position,
                        "heading": tug.heading,
                        "type": "tug_towing" if tug.status == "Towing" else "tug",
                        "ac_id": tug.assigned_ac.id if tug.status == "Towing" and tug.assigned_ac else "",
                        "battery": getattr(tug, "battery", 100),
                        "is_charging": getattr(tug, "is_charging", False),
                        "status2": getattr(tug, "status2", ""),
                    }

            escape_pressed = map_running(map_properties, current_states, t)
            timer.sleep(vis_speed)

        # === 1A. CONTINUOUS SCHEDULING (every spawn_interval seconds) ===
        if t - last_scheduled_time >= spawn_interval:
            # It is randomly an Arrival or Departure!
            ac_type = random.choice(["A", "D"])

            # Initialize the aircraft. scheduled_time is 't'.
            # The Aircraft class automatically calculates spawntime = t + delay
            new_ac = Aircraft(
                next_ac_id,
                ac_type,
                t,
                None,
                None,
                None,
                nodes_dict,
                mean_delay=mean_delay,
                variance_delay=variance_delay,
            )
            unspawned_aircraft.append(new_ac)

            print(
                f"[{t}] Scheduled AC {new_ac.id} ({new_ac.type}). "
                f"Delay: {new_ac.stochastic_delay}s. Spawntime: {new_ac.spawntime}"
            )

            last_scheduled_time = t
            next_ac_id += 1

        # === 1B. DELAYED SPAWNING (Checks if spawntime is reached) ===
        for ac in unspawned_aircraft[:]:
            if t >= ac.spawntime:
                gate_id = airport_manager.get_free_gate()

                # We wait until a gate is actually free before officially spawning it
                if gate_id is None:
                    continue

                ac.gate = gate_id
                ac.sim_mode = sim_mode
                ac.waypoint_queue = []

                if sim_mode == "TUG":
                    if ac.type == "A":
                        airport_manager.gates[gate_id].reserve(ac.id)
                        ac.coupling_zone = airport_manager.arrival_coupling_zones[0]
                        ac.start = airport_manager.arrival_runway_starts[0]
                        ac.goal = airport_manager.arrival_runway_ends[0]
                        ac.waypoint_queue = [ac.coupling_zone]
                        ac.status = "Taxiing"
                    elif ac.type == "D":
                        airport_manager.gates[gate_id].occupy(ac.id)
                        ac.decoupling_zone = airport_manager.departure_decoupling_zones[0]
                        ac.runway_end = airport_manager.departure_runway_ends[0]
                        ac.start = ac.gate
                        ac.goal = ac.gate
                        ac.waypoint_queue = [airport_manager.departure_runway_starts[0], ac.runway_end]
                        ac.status = "AtGate"

                elif sim_mode == "BASELINE":
                    if ac.type == "A":
                        airport_manager.gates[gate_id].reserve(ac.id)
                        ac.start = airport_manager.arrival_runway_starts[0]
                        ac.goal = airport_manager.arrival_runway_ends[0]
                        ac.waypoint_queue = [airport_manager.arrival_coupling_zones[0], ac.gate]
                        ac.status = "Taxiing"
                    elif ac.type == "D":
                        airport_manager.gates[gate_id].occupy(ac.id)
                        ac.decoupling_zone = airport_manager.departure_decoupling_zones[0]
                        ac.runway_end = airport_manager.departure_runway_ends[0]
                        ac.start = ac.gate
                        ac.goal = ac.decoupling_zone
                        ac.waypoint_queue = [airport_manager.departure_runway_starts[0], ac.runway_end]
                        ac.status = "Taxiing"

                ac.position = nodes_dict[ac.start]["xy_pos"]
                aircraft_lst.append(ac)
                unspawned_aircraft.remove(ac)
                print(
                    f"[{t}] AC {ac.id} ({ac.type}) finished delay and spawned at gate {ac.gate}. Status: {ac.status}."
                )

        # === 1.5 DISPATCHER SECTION ===
        # Only run dispatcher logic if Tugs are enabled!
        if sim_mode == "TUG":
            fleet_manager.assign_tugs(aircraft_lst, tug_lst, t)  # Centralized Dispatching

        # === 2. PLANNING SECTION ===
        if planner == "Independent":
            run_independent_planner(aircraft_lst, tug_lst, nodes_dict, edges_dict, heuristics, t)
        elif planner == "Prioritized":
            run_prioritized_planner(aircraft_lst, tug_lst, nodes_dict, edges_dict, heuristics, t, tug_comm_dist_val)
        elif planner == "CBS":
            run_CBS()
        else:
            raise Exception("Planner:", planner, "is not defined.")

        # === 3. MOVEMENT & CLEANUP SECTION ===

        # --- GARBAGE COLLECTION ---
        # Forcefully remove aircraft that have reached their final physical destinations
        # so they release their gates and free up the tugs!
        for ac in aircraft_lst:
            if ac.status != "Removed":
                # Arrivals are done when they physically reach their assigned gate
                if ac.type == "A" and ac.position == nodes_dict[ac.gate]["xy_pos"]:
                    ac.status = "Removed"
                    print(f"[{t}] SUCCESS: AC {ac.id} (Arrival) towed to Gate {ac.gate}. Flow complete!")

                # Departures are done when they physically reach the end of the runway
                if ac.type == "D" and hasattr(ac, "runway_end") and ac.position == nodes_dict[ac.runway_end]["xy_pos"]:
                    ac.status = "Removed"
                    print(f"[{t}] SUCCESS: AC {ac.id} (Departure) reached Runway End. Flow complete!")

        # Clean up aircraft that have finished their flow and release their gates!
        for ac in aircraft_lst:
            if ac.status == "Removed":
                if ac.gate in airport_manager.gates:
                    airport_manager.gates[ac.gate].release()
                    print(f"[{t}] AC {ac.id} removed from simulation. Gate {ac.gate} released.")

                # --- Save KPIs before deleting the aircraft! ---
                completed_kpis.append(
                    {
                        "Flight ID": ac.id,
                        "Type": ac.type,
                        "Total Taxi Time (s)": round(ac.total_taxi_time, 2),
                        "Engine-On Time (s)": round(ac.engine_on_time, 2),
                        "Wait Time for Tug (s)": round(getattr(ac, "wait_for_tug_time", 0.0), 2),  # NEW KPI
                    }
                )

        # Remove them from the active tracking list
        aircraft_lst = [ac for ac in aircraft_lst if ac.status != "Removed"]

        # Move the active agents and update their timers
        for ac in aircraft_lst:
            ac.update_kpis(dt)  # Update the KPI clocks
            if ac.status == "Taxiing":
                ac.move(dt, t)

        for tug in tug_lst:
            tug.update_kpis(dt)  # UPDATE TUG KPIs
            if tug.status == "Waiting" and hasattr(tug, "check_aircraft_arrived"):
                tug.check_aircraft_arrived(t)  # Couple immediately when aircraft arrives
            if tug.status in ["Dispatching", "Towing", "Returning"]:
                tug.move(dt, t)

        t = t + dt

    # === Compile Final Results ===
    tug_kpis = []
    for tug in tug_lst:
        tug_kpis.append(
            {
                "Tug ID": tug.id,
                "Active Towing Time (s)": round(getattr(tug, "towing_time", 0.0), 2),
                "Deadhead Time (s)": round(getattr(tug, "dispatch_time", 0.0) + getattr(tug, "return_time", 0.0), 2),
            }
        )

    throughput_per_hour = (len(completed_kpis) / sim_time) * 3600 if sim_time > 0 else 0

    return {"aircraft": completed_kpis, "tugs": tug_kpis, "throughput": throughput_per_hour}


if __name__ == "__main__":
    # =============================================================================
    # Interactive mode: collect user input then run the simulation
    # =============================================================================

    # --- PHASE 4: TOGGLE SIMULATION MODE ---
    print("\n--- Simulation Configuration ---")
    mode_input = input("Select simulation mode (1 for Electric Tugs, 2 for Baseline Engine-On): ")
    if mode_input.strip() == "2":
        sim_mode = "BASELINE"
        number_of_tugs = 0
        tug_speed_val = 0.0
    else:
        sim_mode = "TUG"
        number_of_tugs = 5
        tug_speed_val = 1.0

    # User visualization speed input
    user_speed = input(
        "\nEnter visualization playback speed (e.g., 1 for normal, 2 for fast, 3 for slow, 4 for very fast): "
    )
    try:
        if float(user_speed) == 2:
            vis_speed = 0.05  # Fast speed
        elif float(user_speed) == 3:
            vis_speed = 0.2  # Slow speed
        elif float(user_speed) == 4:
            vis_speed = 0.001  # Very fast speed
        else:
            vis_speed = 0.1  # Default to normal speed
    except ValueError:
        print("Invalid input. Defaulting to 1.0x speed.")
        vis_speed = 0.1

    config = {
        "mode": sim_mode,
        "spawn_interval": 10.0,
        "number_of_tugs": number_of_tugs,
        "mean_delay": 2.0,
        "variance_delay": 1.0,
        "tug_speed": tug_speed_val,
        "tug_comm_dist": tug_comm_dist,
        "simulation_time": simulation_time,
        "visualization": visualization,
        "visualization_speed": vis_speed,
    }

    # Fetch results dictionary instead of direct list
    results = run_airport_simulation(config)
    completed_kpis = results["aircraft"]

    # =============================================================================
    # 2. Implement analysis of output data here
    # =============================================================================
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE - KPI REPORT")
    print("=" * 50)

    if completed_kpis:
        # Convert our list of dictionaries into a clean Pandas DataFrame
        df_kpis = pd.DataFrame(completed_kpis)
        print("\n--- Aircraft KPIs ---")
        print(df_kpis.to_string(index=False))

        # Calculate Averages
        avg_total_taxi = df_kpis["Total Taxi Time (s)"].mean()
        avg_engine_on = df_kpis["Engine-On Time (s)"].mean()
        avg_wait_time = df_kpis.get("Wait Time for Tug (s)", pd.Series([0])).mean()

        # Calculate fuel/emissions reduction percentage
        if avg_total_taxi > 0:
            savings_pct = ((avg_total_taxi - avg_engine_on) / avg_total_taxi) * 100
        else:
            savings_pct = 0.0

        print("-" * 50)
        print(f"System Throughput:       {results['throughput']:.2f} aircraft/hr")
        print(f"Average Total Taxi Time: {avg_total_taxi:.2f} seconds")
        print(f"Average Engine-On Time:  {avg_engine_on:.2f} seconds")
        print(f"Average Wait for Tug:    {avg_wait_time:.2f} seconds")
        print(f"Engine-On Time Reduced:  {savings_pct:.1f}%")

        # Display Tug Fleet KPIs if applicable
        if results["tugs"]:
            df_tugs = pd.DataFrame(results["tugs"])
            print("\n--- Tug Fleet KPIs ---")
            print(df_tugs.to_string(index=False))

        print("=" * 50)
    else:
        print("No aircraft completed their routes before the simulation ended.")
        print("Try increasing the 'simulation_time' parameter at the top of the script!")
