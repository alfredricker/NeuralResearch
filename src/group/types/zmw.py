from src.group.base import Group
from src.group.base import InputPortSpec

# GROUP FOR FORMING ZMW GROUPS WITHIN A REGION

class ZMWGroup(Group):
    def __init__(self, n: int, theta: float, group_index: int):
        self.group_type = 'zmw'
        super().__init__(n, theta, group_index, self.group_type) # creates neurons for the group

    def expected_input_ports(self) -> dict[str, InputPortSpec]:
        return {
            "ff_from_m": InputPortSpec(source_type="m="),
        }