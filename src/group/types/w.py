from src.group.base import Group
from src.group.base import InputPortSpec
from src.neuron.edge_topology import RingTopology

# GROUP FOR FORMING W GROUPS WITHIN A REGION

class WGroup(Group):
    def __init__(self, n: int, theta: float, group_index: int | None = None):
        self.group_type = 'w'
        super().__init__(n, theta, self.group_type, group_index, recurrent_topology=RingTopology()) # creates neurons for the group

    def expected_input_ports(self) -> dict[str, InputPortSpec]:
        return {
            "ff_from_omega": InputPortSpec(source_type="m="),
        }