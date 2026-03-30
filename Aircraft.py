import math
import random
from single_agent_planner import simple_single_agent_astar


class Aircraft(object):
    """Aircraft class matching the TU Delft Formal Agent Model Concept of Operations."""

    def __init__(
        self,
        flight_id,
        a_d,
        scheduled_time,
        gate_node,
        runway_node,
        coupling_node,
        nodes_dict,
        mean_delay=2.0,
        variance_delay=1.0,
        speed=1.0,
    ):
        # 1. Attributes
        self.id = flight_id
        self.type = a_d  # "A" for arrival, "D" for departure
        self.engine_on_taxi_speed = speed
        self.speed = self.engine_on_taxi_speed
        self.nodes_dict = nodes_dict

        # Scheduling & Stochastic Delay
        self.scheduled_time = scheduled_time
        self.stochastic_delay = max(
            0,
            random.gauss(mean_delay, math.sqrt(max(0.0, variance_delay))),
        )
        self.spawntime = self.scheduled_time + self.stochastic_delay

        # Layout Assignments (Dynamically assigned by run_me.py via AirportResourceManager)
        self.gate = gate_node
        self.runway = runway_node
        self.coupling_zone = coupling_node
        self.start = None
        self.goal = None

        # 2. States & Routes
        self.status = None  # "Taxiing", "Coupled", "AtGate", or "Removed"
        self.path_to_goal = []
        self.waypoint_queue = []
        self.from_to = [0, 0]
        self.heading = 0
        self.position = (0, 0)

        # KPI Trackers
        self.tug_assigned = False
        self.is_being_towed = False
        self.engine_on_time = 0.0
        self.total_taxi_time = 0.0
        self.wait_for_tug_time = 0.0
        self.actual_start_time = None
        self.actual_end_time = None

    def update_kpis(self, dt):
        """Updates time tracking based on the current state."""
        if self.status == "Taxiing":
            self.engine_on_time += dt
            self.total_taxi_time += dt
        elif self.status == "Coupled":
            self.total_taxi_time += dt
            # Arrival planes waiting at the coupling zone
            if not self.is_being_towed and getattr(self, "sim_mode", "TUG") == "TUG":
                self.wait_for_tug_time += dt
        elif self.status == "AtGate":
            # Departure planes waiting at the gate
            if not self.is_being_towed and getattr(self, "sim_mode", "TUG") == "TUG":
                self.wait_for_tug_time += dt

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

    def move(self, dt, t):
        """Handles movement under own engine power (Taxiing state)."""
        if self.status != "Taxiing" or len(self.path_to_goal) == 0:
            return

        to_node = self.path_to_goal[0][0]
        xy_to = self.nodes_dict[to_node]["xy_pos"]

        self.get_heading(self.position, xy_to)

        x = xy_to[0] - self.position[0]
        y = xy_to[1] - self.position[1]
        dist_to_target = math.hypot(x, y)
        distance_to_move = self.speed * dt

        # If we are close enough to hit the target in this timestep
        if dist_to_target <= distance_to_move + 0.001:
            self.position = xy_to

            # Check if this node is our current goal
            if self.position == self.nodes_dict[self.goal]["xy_pos"]:

                # WAYPOINT LOGIC: Do we have more waypoints to hit in this taxiing phase?
                if len(self.waypoint_queue) > 0 and self.status == "Taxiing":
                    self.start = self.goal
                    self.goal = self.waypoint_queue.pop(0)
                    self.path_to_goal = []  # Clear path to trigger A* replan immediately
                    return

                # FINAL DESTINATION LOGIC
                sim_mode = getattr(self, "sim_mode", "TUG")

                if sim_mode == "TUG":
                    if self.type == "A" and self.goal == self.coupling_zone:
                        self.status = "Coupled"
                        self.speed = 0
                        print(f"[{round(t+dt, 2)}] AC {self.id} reached coupling zone. Engines OFF.")
                    elif self.type == "D" and self.goal == getattr(self, "runway_end", -1):
                        self.status = "Removed"

                elif sim_mode == "BASELINE":
                    if self.type == "A" and self.goal == self.gate:
                        self.status = "Removed"
                    elif self.type == "D" and self.goal == getattr(self, "runway_end", -1):
                        self.status = "Removed"
            else:
                self.path_to_goal = self.path_to_goal[1:]
                if len(self.path_to_goal) > 0:
                    self.from_to = [self.from_to[1], self.path_to_goal[0][0]]
        else:
            x_normalized = x / dist_to_target
            y_normalized = y / dist_to_target
            posx = round(self.position[0] + x_normalized * distance_to_move, 3)
            posy = round(self.position[1] + y_normalized * distance_to_move, 3)
            self.position = (posx, posy)

    def plan_independent(self, nodes_dict, edges_dict, heuristics, t):
        """Plans path to current goal using A*."""
        if self.status == "Taxiing" and len(self.path_to_goal) == 0:
            success, path = simple_single_agent_astar(nodes_dict, self.start, self.goal, heuristics, t)
            if success:
                self.path_to_goal = path[1:]
                if len(self.path_to_goal) > 0:
                    self.from_to = [path[0][0], self.path_to_goal[0][0]]
                print(f"Path AC {self.id}: {path}")
            else:
                print(f"No solution found for AC {self.id}")
