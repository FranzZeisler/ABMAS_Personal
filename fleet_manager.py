"""
fleet_manager.py – Centralised Fleet Manager for Electric Tug Dispatch.

Implements the four-step assignment pipeline:
    T1 – Build priority queue of waiting aircraft (sorted by stochastic delay)
    T2 – Compute heuristic distance from each available tug to the aircraft pickup node
    T3 – Battery feasibility filter (full round-trip distance check)
    T4 – Select closest viable tug and dispatch it, transmitting both the
         target node (pickup/coupling location) and destination node (drop-off/
         decoupling location) to the tug at dispatch time.

The manager also continuously monitors:
    - Coupling zone and gate nodes for aircraft waiting for a tug
    - Each tug's location and current charge state
"""


class FleetManager:
    """
    Monitors coupling zones and tug states each timestep, then dispatches tugs
    to waiting aircraft following the T1->T2->T3->T4 pipeline.
    """

    def __init__(self, heuristics, nodes_dict):
        """
        Parameters
        ----------
        heuristics : dict
            Pre-computed shortest-path distance table {from_node: {to_node: distance}}.
        nodes_dict : dict
            Full airport node dictionary used to identify coupling/gate zones
            for active monitoring.
        """
        self.heuristics = heuristics
        self.nodes_dict = nodes_dict

        # Pre-index node types so monitoring does not re-scan every timestep
        self.coupling_zone_nodes = {nid for nid, d in nodes_dict.items() if d.get("type") == "couple"}
        self.decoupling_zone_nodes = {nid for nid, d in nodes_dict.items() if d.get("type") == "decouple"}
        self.gate_nodes = {nid for nid, d in nodes_dict.items() if d.get("type") == "gate"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_tugs(self, aircraft_lst, tug_lst, t):
        """
        Run monitoring + full T1-T4 assignment cycle each simulation timestep.
        Call once per timestep *before* the planning / movement sections.

        Parameters
        ----------
        aircraft_lst : list[Aircraft]
            All aircraft currently active in the simulation.
        tug_lst : list[Tug]
            All tugs in the fleet.
        t : float
            Current simulation time (used for logging).
        """
        # -- Continuous Monitoring -----------------------------------------
        self._monitor_state(aircraft_lst, tug_lst, t)

        # -- T1: Priority Queue --------------------------------------------
        # Collect aircraft that are at a monitored waiting node and have not
        # yet been assigned a tug. Sort descending by stochastic_delay so the
        # most delayed aircraft is served first.
        waiting_aircraft = self._get_waiting_aircraft(aircraft_lst)
        if not waiting_aircraft:
            return  # Nothing to assign this timestep

        waiting_aircraft = self._sort_by_priority(waiting_aircraft)

        for ac in waiting_aircraft:
            # -- T2: Distance Calculation ----------------------------------
            # Resolve pickup (coupling/gate) and drop-off (gate/decoupling) nodes.
            pickup_node = self._get_pickup_node(ac)
            dropoff_node = self._get_dropoff_node(ac)

            if pickup_node is None or dropoff_node is None:
                continue  # Aircraft not fully configured yet - skip

            idle_tugs = [tug for tug in tug_lst if tug.status == "Idle"]
            if not idle_tugs:
                break  # No idle tugs remain; nothing more to assign this tick

            # Compute distance from the aircraft's pickup node to every idle
            # tug (aircraft -> tug direction, per spec). Result is sorted
            # ascending so the closest tug appears first.
            tug_distances = self._compute_distances(idle_tugs, pickup_node)

            # -- T3: Battery Feasibility Filter ----------------------------
            # Retain only tugs whose battery can sustain the full round-trip:
            #     tug_start -> pickup_node -> dropoff_node -> tug.base_node
            feasible_tugs = self._filter_by_battery(tug_distances, pickup_node, dropoff_node)

            if not feasible_tugs:
                print(
                    f"[{round(t, 2)}] Fleet Manager: No battery-feasible tug "
                    f"available for AC {ac.id}. Will retry next timestep."
                )
                continue

            # -- T4: Select and Dispatch -----------------------------------
            # Transmit both the target node (pickup) and destination node
            # (drop-off) to the selected tug.
            self._dispatch(feasible_tugs, ac, pickup_node, dropoff_node, t)

    # ------------------------------------------------------------------
    # Continuous Monitoring
    # ------------------------------------------------------------------

    def _monitor_state(self, aircraft_lst, tug_lst, t):
        """
        Actively monitors coupling/gate nodes for waiting aircraft and logs
        each tug's current location and charge state. Called every timestep
        before the assignment pipeline runs.
        """
        # --- Monitor coupling zones, departure gates, and taxiing arrivals ---
        waiting_at_nodes = {}
        taxiing_unassigned = []
        for ac in aircraft_lst:
            if getattr(ac, "tug_assigned", False):
                continue  # Already handled
            if ac.status == "Coupled" and getattr(ac, "coupling_zone", None) in self.coupling_zone_nodes:
                waiting_at_nodes[ac.coupling_zone] = ac
            elif ac.status == "AtGate" and getattr(ac, "gate", None) in self.gate_nodes:
                waiting_at_nodes[ac.gate] = ac
            elif ac.status == "Taxiing" and ac.type == "A":
                taxiing_unassigned.append(ac)

        if waiting_at_nodes:
            node_summary = ", ".join(f"node {node} (AC {ac.id})" for node, ac in waiting_at_nodes.items())
            print(f"[{round(t, 2)}] Monitor | Waiting aircraft at: {node_summary}")

        if taxiing_unassigned:
            ac_summary = ", ".join(f"AC {ac.id}" for ac in taxiing_unassigned)
            print(f"[{round(t, 2)}] Monitor | Taxiing arrivals awaiting tug pre-dispatch: {ac_summary}")

        # --- Monitor tug locations and charge states ---
        # for tug in tug_lst:
        # charge_pct = tug.battery
        # print(
        #    f"[{round(t, 2)}] Monitor | Tug {tug.id} "
        #    f"@ node {tug.start} | status: {tug.status} | "
        #    f"charge: {tug.battery:.1f} "
        #    f"({charge_pct:.0f}%)"
        # )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_waiting_aircraft(self, aircraft_lst):
        """
        Return aircraft that need a tug assigned and have not yet been assigned one.
        This includes:
          - Arrivals ("A") that are actively taxiing: tug is dispatched immediately
            on spawn so it can travel to the coupling zone and be ready on arrival.
          - Arrivals ("A") already waiting at the coupling zone (Coupled status).
          - Departures ("D") waiting at their gate (AtGate status).
        """
        waiting = []
        for ac in aircraft_lst:
            if getattr(ac, "tug_assigned", False):
                continue
            if (
                ac.type == "A"
                and ac.status in ["Taxiing", "Coupled"]
                and getattr(ac, "coupling_zone", None) in self.coupling_zone_nodes
            ):
                waiting.append(ac)
            elif ac.type == "D" and ac.status == "AtGate" and getattr(ac, "gate", None) in self.gate_nodes:
                waiting.append(ac)
        return waiting

    def _sort_by_priority(self, waiting_aircraft):
        """T1 - Sort by stochastic_delay descending (highest delay = highest priority)."""
        return sorted(waiting_aircraft, key=lambda ac: ac.stochastic_delay, reverse=True)

    def _get_pickup_node(self, ac):
        """The node the tug must travel to in order to couple with the aircraft."""
        if ac.type == "A":
            return getattr(ac, "coupling_zone", None)
        else:  # "D"
            return getattr(ac, "gate", None)

    def _get_dropoff_node(self, ac):
        """The node the tug must deliver the aircraft to."""
        if ac.type == "A":
            return getattr(ac, "gate", None)
        else:  # "D"
            return getattr(ac, "decoupling_zone", None)

    def _compute_distances(self, idle_tugs, pickup_node):
        """
        T2 - For a given aircraft pickup node, compute the distance from that
        node to each idle tug's current position (aircraft -> tug direction,
        per spec). Returns a list of (distance, tug) tuples sorted ascending
        by distance so the closest tug is always first.
        Tugs with no reachable heuristic path are excluded.
        """
        results = []
        for tug in idle_tugs:
            try:
                # Direction: pickup_node -> tug.start (aircraft to tug, per spec)
                dist = self.heuristics[pickup_node][tug.start]
                results.append((dist, tug))
            except KeyError:
                pass  # No path exists - skip this tug
        results.sort(key=lambda x: x[0])
        return results

    def _filter_by_battery(self, tug_distances, pickup_node, dropoff_node):
        """
        T3 - Remove tugs whose battery cannot sustain the full round-trip:
            tug_start -> pickup_node -> dropoff_node -> tug.base_node
        Returns a filtered, distance-sorted list of (distance, tug) tuples.
        """
        feasible = []
        for dist, tug in tug_distances:
            if tug.can_complete_roundtrip(pickup_node, dropoff_node, self.heuristics):
                feasible.append((dist, tug))
            # else:
            # print(
            #    f"    [T3 Battery] Tug {tug.id} filtered out - "
            #    f"charge {tug.battery:.1f} "
            #    f"insufficient for round-trip."
            # )
        return feasible

    def _dispatch(self, feasible_tugs, ac, pickup_node, dropoff_node, t):
        """
        T4 - Select the closest feasible tug and dispatch it, explicitly
        transmitting both:
            - target node      (pickup_node   : where to go couple with the aircraft)
            - destination node (dropoff_node  : where to deliver the aircraft)
        Both are stored on the tug so it never needs to query the aircraft
        directly for routing decisions.
        """
        min_dist, best_tug = feasible_tugs[0]

        # Transmit both nodes to the tug at dispatch time
        best_tug.status = "Dispatching"
        best_tug.assigned_ac = ac
        best_tug.goal = pickup_node  # Target node (coupling location)
        best_tug.dispatched_dropoff = dropoff_node  # Destination node (decoupling location)
        best_tug.path_to_goal = []  # Force A* replan on next planning tick

        ac.tug_assigned = True  # Prevent double-assignment

        print(
            f"[{round(t, 2)}] Fleet Manager dispatched Tug {best_tug.id} -> AC {ac.id} | "
            f"target node (pickup): {pickup_node} | "
            f"destination node (drop-off): {dropoff_node} | "
            f"distance: {min_dist:.2f} | "
            f"charge: {best_tug.battery:.1f}"
        )
