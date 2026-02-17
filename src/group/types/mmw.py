from src.group.base import Group
from src.group.base import InputPortSpec

# GROUP FOR FORMING MMW GROUPS WITHIN A REGION

class MMWGroup(Group):
    def __init__(self, n: int, theta: float, group_index: int):
        self.group_type = 'mmw'
        super().__init__(n, theta, group_index, self.group_type) # creates neurons for the group

    def expected_input_ports(self) -> dict[str, InputPortSpec]:
        return {
            "ff_from_omega": InputPortSpec(source_type="omega="),
        }