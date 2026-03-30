class Gate:
    """Represents a physical gate with state tracking."""

    def __init__(self, node_id, xy_pos):
        self.node_id = node_id
        self.xy_pos = xy_pos
        self.status = "Free"  # States: "Free", "Reserved", "Occupied"
        self.assigned_aircraft_id = None

    def reserve(self, aircraft_id):
        if self.status == "Free":
            self.status = "Reserved"
            self.assigned_aircraft_id = aircraft_id
            return True
        return False

    def occupy(self, aircraft_id):
        self.status = "Occupied"
        self.assigned_aircraft_id = aircraft_id

    def release(self):
        self.status = "Free"
        self.assigned_aircraft_id = None


class AirportResourceManager:
    """Parses the node dictionary to manage gates and special zones based on exact types."""

    def __init__(self, nodes_dict):
        self.nodes_dict = nodes_dict
        self.gates = {}

        # Explicit Zones
        self.arrival_runway_starts = []
        self.arrival_runway_ends = []
        self.departure_runway_starts = []
        self.departure_runway_ends = []
        self.arrival_coupling_zones = []
        self.departure_decoupling_zones = []

        self._initialize_infrastructure()

    def _initialize_infrastructure(self):
        for node_id, data in self.nodes_dict.items():
            node_type = data.get("type", "")

            if node_type == "gate":
                self.gates[node_id] = Gate(node_id, data["xy_pos"])
            elif node_type == "rwy_a_start":
                self.arrival_runway_starts.append(node_id)
            elif node_type == "rwy_a_end":
                self.arrival_runway_ends.append(node_id)
            elif node_type == "rwy_d_start":
                self.departure_runway_starts.append(node_id)
            elif node_type == "rwy_d_end":
                self.departure_runway_ends.append(node_id)
            elif node_type == "couple":
                self.arrival_coupling_zones.append(node_id)
            elif node_type == "decouple":
                self.departure_decoupling_zones.append(node_id)

    def get_free_gate(self):
        for gate_id, gate_obj in self.gates.items():
            if gate_obj.status == "Free":
                return gate_id
        return None
