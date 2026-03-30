import math

def run_prioritized_planner(aircraft_lst, tug_lst, nodes_dict, edges_dict, heuristics, t, comm_dist):
    for ac in aircraft_lst:
        if ac.status == "Taxiing" and len(ac.path_to_goal) == 0:
            ac.plan_independent(nodes_dict, edges_dict, heuristics, t)

    for i, tug in enumerate(tug_lst):
        # Communicate between tugs (pairwise with later-indexed tugs)
        if i < len(tug_lst) - 1:
            for other_tug in tug_lst[i+1:]:
                dist = math.hypot(tug.position[0] - other_tug.position[0],
                                  tug.position[1] - other_tug.position[1])
                if dist <= comm_dist:
                    # Always send latest plan (including empty) so stale constraints are removed.
                    tug.receive_constraints(other_tug.id, other_tug.path_to_goal)
                    other_tug.receive_constraints(tug.id, tug.path_to_goal)

        if tug.status in ["Dispatching", "Towing", "Returning"] and len(tug.path_to_goal) == 0:
            tug.plan_independent(nodes_dict, edges_dict, heuristics, t)

    