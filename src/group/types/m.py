from src.group.base import Group
from src.group.base import InputPortSpec
from src.neuron.edge_topology import RingTopology

# GROUP FOR FORMING ABSTRACT MODELS WITHIN A REGION

class MGroup(Group):
    def __init__(self, n: int, theta: float, group_index: int):
        self.group_type = 'm'
        super().__init__(n, theta, group_index, self.group_type, recurrent_topology=RingTopology()) # creates neurons for the group

    def expected_input_ports(self) -> dict[str, InputPortSpec]:
        # M-like groups typically receive feedforward sensory drive and optional
        # contextual or recurrently external modulatory streams.
        return {
            "ff_from_omega": InputPortSpec(source_type="omega="),
            "gating_from_w": InputPortSpec(source_type="w="),
            #"fb_from_z": InputPortSpec(source_type="z="),
            "lateral_from_m": InputPortSpec(source_type="m|")
        }