from src.cortical.group.base import Group
from src.cortical.group.base import InputPortSpec

# GROUP FOR FORMING Z GROUPS WITHIN A REGION

class ZGroup(Group):
    def __init__(self, n: int, theta: float, group_index: int | None = None):
        self.group_type = 'z'
        super().__init__(n, theta, self.group_type, group_index) # creates neurons for the group

    def expected_input_ports(self) -> dict[str, InputPortSpec]:
        return {
            "ff_from_m": InputPortSpec(source_type="m="),
            "ff_from_w": InputPortSpec(source_type="w="),
        }