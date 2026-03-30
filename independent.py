def run_independent_planner(aircraft_lst, tug_lst, nodes_dict, edges_dict, heuristics, t):
    for ac in aircraft_lst:
        if ac.status == "Taxiing" and len(ac.path_to_goal) == 0:
            ac.plan_independent(nodes_dict, edges_dict, heuristics, t)

    for tug in tug_lst:
        if tug.status in ["Dispatching", "Towing", "Returning"] and len(tug.path_to_goal) == 0:
            tug.plan_independent(nodes_dict, edges_dict, heuristics, t)
