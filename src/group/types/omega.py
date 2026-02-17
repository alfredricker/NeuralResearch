from src.group.base import Group
from src.group.base import InputPortSpec

# GROUP FOR FORMING OMEGA GROUPS WITHIN A REGION

class OmegaGroup(Group):
    def __init__(self, n: int, theta: float, group_index: int):
        self.group_type = 'omega'
        super().__init__(n, theta, group_index, self.group_type) # creates neurons for the group

    def expected_input_ports(self) -> dict[str, InputPortSpec]:
        return {
            "ff_from_z": InputPortSpec(source_type="z-"),
            "fb_from_m": InputPortSpec(source_type="m+"),
        }