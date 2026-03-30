import math
from single_agent_planner import simple_single_agent_astar
from collections import Counter


class Tug(object):
    """Tug class that actively dispatches and tows passive aircraft."""

    # Discharge rates (units per second)
    DISCHARGE_STATIONARY = 0.1  # Low
    DISCHARGE_MOVING = 0.5  # Mid
    DISCHARGE_TOWING = 2.0  # High
    CHARGE_RATE = 3.0  # Recovery when charging

    def __init__(self, tug_id, start_node, nodes_dict, speed=1.0):
        self.id = tug_id
        self.speed = speed
        self.nodes_dict = nodes_dict
        self.start = start_node
        self.goal = start_node
        self.base_node = start_node  # NEW: Remember where to return to charge/idle
        self.position = nodes_dict[start_node]["xy_pos"]  # tuple (x, y)

        # Route & State related mapped to Concept of Ops (Idle=Charging, Dispatching=Dispatched, Towing=Servicing)
        self.status = "Idle"
        self.status2 = "stationary"
        self.assigned_ac = None
        self.path_to_goal = []
        self.from_to = [start_node, start_node]
        self.heading = 0

        self.vertex_constraints_set = (
            Counter()
        )  # Vertex constraints received from other tugs. Reference-counted by constraint key.
        self.edge_constraints_set = Counter()  # Edge constraints, also reference-counted.
        self._constraints_by_source = {}  # Track constraints by source tug id so updates can replace stale constraints.

        # Dispatch instructions transmitted by the Fleet Manager
        self.dispatched_dropoff = None  # Drop-off node assigned at dispatch time (T4)

        # Battery model (Unified)
        self.battery = 100.0  # Current charge level (Percentage 0-100)
        self.battery_maxcapacity = 100.0

        # Flags for visualizer and status checks
        self.is_towing = False
        self.is_charging = False

        # KPI Trackers
        self.towing_time = 0.0
        self.dispatch_time = 0.0
        self.return_time = 0.0

    def update_kpis(self, dt):
        """Updates internal KPI timers based on tug status."""
        if self.status == "Towing":
            self.towing_time += dt
        elif self.status == "Dispatching":
            self.dispatch_time += dt
        elif self.status == "Returning":
            self.return_time += dt

    def get_heading(self, xy_start, xy_next):
        """Determines heading based on start and end xy position."""
        if xy_start[0] == xy_next[0]:
            if xy_start[1] > xy_next[1]:
                self.heading = 180
            elif xy_start[1] < xy_next[1]:
                self.heading = 0
        elif xy_start[1] == xy_next[1]:
            if xy_start[0] > xy_next[0]:
                self.heading = 90
            elif xy_start[0] < xy_next[0]:
                self.heading = 270

    def check_charging_status(self):
        """Checks if the tug is stationary on a charging node."""
        # 1. Get the ID of the node the tug is currently at
        # If stationary, it has reached its 'to_node'
        current_node_id = self.from_to[1]

        # 2. Get the type of that node from the dictionary
        node_type = self.nodes_dict[current_node_id].get("type")

        # 3. Logic: Must be stationary AND on a charger
        if self.status2 == "stationary" and node_type == "charging_station":
            self.is_charging = True
        else:
            self.is_charging = False

    def update_battery(self, dt):
        """Calculates battery drain/gain for the current time step."""
        # determining if tug is at charging station
        self.check_charging_status()

        # 1. Determine Rate
        if self.is_charging:
            current_rate = -self.CHARGE_RATE  # Negative drain = charging
        elif self.status2 == "moving" and self.status == "Towing":
            current_rate = self.DISCHARGE_TOWING
        elif self.status2 == "moving":
            current_rate = self.DISCHARGE_MOVING
        else:
            current_rate = self.DISCHARGE_STATIONARY

        # 2. Apply Change
        self.battery -= current_rate * dt

        # 3. Clamp between 0 and 100
        self.battery = max(0.0, min(self.battery_maxcapacity, self.battery))

    def move(self, dt, t):
        """Moves the tug, and syncs the attached aircraft's position if towing."""

        # --- FIX: ZERO-DISTANCE DISPATCH CHECK ---
        # If we are supposed to be moving but have no path, check if we are already at the goal.
        if self.status in ["Dispatching", "Towing", "Returning"] and len(self.path_to_goal) == 0:
            if self.position == self.nodes_dict[self.goal]["xy_pos"]:
                self._handle_goal_reached(t)
                return

        # Original gatekeeper (modified to allow the check above to run first)
        if self.status not in ["Dispatching", "Towing", "Returning"] or len(self.path_to_goal) == 0:
            return

        to_node = self.path_to_goal[0][0]
        xy_to = self.nodes_dict[to_node]["xy_pos"]

        self.get_heading(self.position, xy_to)

        # Calculate exact distance to the next node
        x = xy_to[0] - self.position[0]
        y = xy_to[1] - self.position[1]
        dist_to_target = math.hypot(x, y)
        distance_to_move = self.speed * dt

        # If we are close enough to hit the target in this timestep
        if dist_to_target <= distance_to_move + 0.001:
            # Drain battery for the actual distance covered to this node
            self.position = xy_to  # Snap exactly to the node
            self.status2 = "stationary"

            # Sync passive aircraft if towing
            if self.status == "Towing" and self.assigned_ac:
                self.assigned_ac.position = self.position
                self.assigned_ac.heading = self.heading

            # Check if this node is our final destination
            if self.position == self.nodes_dict[self.goal]["xy_pos"]:
                self._handle_goal_reached(t + dt)
            else:
                # Target not reached, shift to the next node in the path
                self.path_to_goal = self.path_to_goal[1:]
                if len(self.path_to_goal) > 0:
                    self.from_to = [self.from_to[1], self.path_to_goal[0][0]]
        elif self.battery > 0:
            # Normal movement step across the map
            x_normalized = x / dist_to_target
            y_normalized = y / dist_to_target
            posx = round(self.position[0] + x_normalized * distance_to_move, 3)
            posy = round(self.position[1] + y_normalized * distance_to_move, 3)
            self.position = (posx, posy)
            # Drain battery proportional to the distance actually moved this step
            self.status2 = "moving"

            # Sync passive aircraft if towing
            if self.status == "Towing" and self.assigned_ac:
                self.assigned_ac.position = self.position
                self.assigned_ac.heading = self.heading
        else:
            return

        # Update battery at the start of every step
        self.update_battery(dt)

    def _handle_goal_reached(self, t):
        """Handles state transitions when the Tug arrives at a target node."""
        t = round(t, 2)

        # Scenario 1: Tug arrived at the pickup node
        if self.status == "Dispatching":
            self.start = self.goal
            self.path_to_goal = []  # Clear path

            ac = self.assigned_ac
            # Couple immediately if:
            #   - arrival aircraft already reached coupling zone ("Coupled"), or
            #   - departure aircraft is at its gate ("AtGate") - it is always there
            # Only arrivals still taxiing trigger the Waiting state.
            if ac.status == "Coupled" or (ac.type == "D" and ac.status == "AtGate"):
                self._start_towing(t)
            else:
                # Arrival aircraft has not yet reached the coupling zone - wait
                self.status = "Waiting"
                print(
                    f"[{t}] Tug {self.id} arrived at pickup node {self.goal} early. Waiting for AC {self.assigned_ac.id}."
                )

        # Scenario 2: Tug arrived at drop-off zone with the Aircraft
        elif self.status == "Towing":
            print(f"[{t}] Tug {self.id} reached destination. Decoupling from AC {self.assigned_ac.id}.")
            self.assigned_ac.is_being_towed = False  # TURN OFF TOWING FLAG

            if self.assigned_ac.type == "A":
                self.assigned_ac.status = "AtGate"
                self.assigned_ac.position = self.nodes_dict[self.goal]["xy_pos"]
                self.assigned_ac.status = "Removed"

            elif self.assigned_ac.type == "D":
                self.assigned_ac.status = "Taxiing"  # Engines back ON!
                self.assigned_ac.speed = self.assigned_ac.engine_on_taxi_speed
                self.assigned_ac.start = self.goal

                # Pop the next waypoint (rwy_d_start) to continue autonomous taxi
                if len(self.assigned_ac.waypoint_queue) > 0:
                    self.assigned_ac.goal = self.assigned_ac.waypoint_queue.pop(0)

                self.assigned_ac.path_to_goal = []  # Force aircraft to plan route

            # --- Phase 5 Setup - Return to Base ---
            self.status = "Returning"
            self.assigned_ac = None
            self.dispatched_dropoff = None  # Clear the transmitted instruction
            self.start = self.goal
            self.goal = self.base_node
            self.path_to_goal = []  # Force A* replan back to charger
            print(f"[{t}] Tug {self.id} returning to base node {self.base_node}.")

        # Scenario 3: Tug arrived back at its charging/base node
        elif self.status == "Returning":
            self.status = "Idle"
            self.start = self.goal
            print(f"[{t}] Tug {self.id} arrived at base node {self.base_node} and is now Idle.")

    def _build_constraints_from_path(self, path):
        """Build vertex and edge constraints from a path"""
        vertex_constraints = set()
        edge_constraints = set()

        for idx, node in enumerate(path):
            loc, timestep = node
            vertex_constraints.add((loc, timestep, False))

            if idx > 0:
                prev_loc, prev_timestep = path[idx - 1]
                # Edge is occupied during [prev_timestep, timestep]
                edge_constraints.add(((prev_loc, loc), (prev_timestep, timestep), False))

        return vertex_constraints, edge_constraints

    def _decrement_constraints(self, counter, constraints):
        """
        Decrease reference counts for a collection of constraints.

        When a count drops to zero or below, remove the constraint entry so that
        membership tests and iteration only see active constraints.
        """
        for constraint in constraints:
            if constraint in counter:
                counter[constraint] -= 1
                if counter[constraint] <= 0:
                    del counter[constraint]

    def remove_constraints_from_source(self, source_tug_id):
        """Remove all currently stored constraints that were received from one tug"""
        previous = self._constraints_by_source.pop(source_tug_id, None)
        if previous is None:
            return

        prev_vertex, prev_edge = previous
        self._decrement_constraints(self.vertex_constraints_set, prev_vertex)
        self._decrement_constraints(self.edge_constraints_set, prev_edge)

    def clear_all_received_constraints(self):
        """Clear all received constraints from all sources"""
        self.vertex_constraints_set.clear()
        self.edge_constraints_set.clear()
        self._constraints_by_source.clear()

    def receive_constraints(self, source_tug_id, path):
        """
        Replace constraints from one source tug using its latest communicated path
        - path format: [(loc, timestep), ...]
        - vertex format: (loc, timestep, positive)
        - edge format: ((from_loc, to_loc), (start_timestep, end_timestep), positive)
        """
        # Remove old constraints from this source first to avoid stale reservations
        # TODO: Add path ID which gets incremented if the other tug did replanning. Then
        # we can only update constraints when the other tug's path is actually updated
        self.remove_constraints_from_source(source_tug_id)

        vertex_constraints, edge_constraints = self._build_constraints_from_path(path)
        self.vertex_constraints_set.update(vertex_constraints)
        self.edge_constraints_set.update(edge_constraints)
        self._constraints_by_source[source_tug_id] = (vertex_constraints, edge_constraints)

    def can_complete_roundtrip(self, pickup_node, dropoff_node, heuristics):
        """
        T3 – Battery Feasibility Check.
        Returns True if the tug's current battery level is sufficient to cover:
            tug_start → pickup_node → dropoff_node → base_node
        """
        try:
            d1 = heuristics[self.start][pickup_node]  # tug → aircraft
            d2 = heuristics[pickup_node][dropoff_node]  # aircraft → drop-off
            d3 = heuristics[dropoff_node][self.base_node]  # drop-off → base

            required_charge = (
                (d1 + d3) * self.DISCHARGE_MOVING + d2 * self.DISCHARGE_TOWING + 10
            )  # 10 so some margin for battery not going below zero
            return self.battery >= required_charge

        except KeyError:
            return False  # Unreachable node – treat as infeasible

    def _start_towing(self, t):
        """Couples with the assigned aircraft and transitions to Towing state."""
        self.status = "Towing"

        # Use the drop-off node transmitted by the Fleet Manager at dispatch time (T4)
        if self.dispatched_dropoff is not None:
            self.goal = self.dispatched_dropoff
        else:
            # Fallback: read directly from aircraft (should not occur in normal operation)
            if self.assigned_ac.type == "A":
                self.goal = self.assigned_ac.gate
            elif self.assigned_ac.type == "D":
                self.goal = getattr(self.assigned_ac, "decoupling_zone", self.goal)

        self.assigned_ac.status = "Coupled"
        self.assigned_ac.is_being_towed = True  # TURN ON TOWING FLAG
        self.assigned_ac.speed = 0
        self.path_to_goal = []  # Force A* replan toward drop-off
        print(f"[{t}] Tug {self.id} coupled with AC {self.assigned_ac.id}. Towing to node {self.goal}.")

    def check_aircraft_arrived(self, t):
        """
        Called each timestep when status is Waiting.
        As soon as the assigned aircraft reaches the coupling zone (status becomes
        Coupled), the tug immediately couples and transitions to Towing.
        """
        if self.assigned_ac is not None and self.assigned_ac.status == "Coupled":
            self._start_towing(round(t, 2))

    def plan_independent(self, nodes_dict, edges_dict, heuristics, t):
        """Plans a path for the tug independently using A*."""
        # NEW: Allow planning if Returning
        if self.status in ["Dispatching", "Towing", "Returning"] and len(self.path_to_goal) == 0:
            success, path = simple_single_agent_astar(nodes_dict, self.start, self.goal, heuristics, t)
            if success:
                self.path_to_goal = path[1:]
                if len(self.path_to_goal) > 0:
                    self.from_to = [path[0][0], self.path_to_goal[0][0]]
            else:
                print(f"No solution found for Tug {self.id}")
