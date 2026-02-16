from src.group.base import Group
from src.group.base import InputPortSpec

class ModelGroup(Group):
    def __init__(self, n: int, theta: float, id: int = 0):
        group_id = f'm{id}'
        super().__init__(n, theta, group_id) # creates neurons for the group

    def expected_input_ports(self) -> dict[str, InputPortSpec]:
        # M-like groups typically receive feedforward sensory drive and optional
        # contextual or recurrently external modulatory streams.
        return {
            "ff_from_omega": InputPortSpec(
                name="ff_from_omega",
                required=True,
                allowed_source_types={"omega"},
                min_bindings=1,
            ),
            "context_from_z": InputPortSpec(
                name="context_from_z",
                required=False,
                allowed_source_types={"z"},
                min_bindings=0,
            ),
            "gating_from_w": InputPortSpec(
                name="gating_from_w",
                required=False,
                allowed_source_types={"w"},
                min_bindings=0,
            ),
        }