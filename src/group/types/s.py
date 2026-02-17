#######################################################################
# SENSORY NEURONS
# These neurons simply receive information from an external domain.
# They do not have any gating mechanisms or hebbian learning rules.
#######################################################################

from src.group.base import Group
from src.group.base import InputPortSpec

# GROUP FOR FORMING SENSORY GROUPS WITHIN A REGION

class SGroup(Group):
    def __init__(self, n: int, theta: float, domain_id: str, group_index: int | None = None):
        self.group_type = 's'
        self.domain_id = domain_id
        super().__init__(n, theta, self.group_type, group_index) # creates neurons for the group

    def expected_input_ports(self) -> dict[str, InputPortSpec]:
        return {
            "ff_from_domain": InputPortSpec(source_type=f"dom_{self.domain_id}"),
        }