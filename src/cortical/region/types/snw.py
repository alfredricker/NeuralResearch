##########################################################################
# SNW stands for Sensory No Where region -- no where neurons are present
##########################################################################

from src.cortical.region.region import Region
from src.cortical.group.types.omega import OmegaGroup
from src.cortical.group.types.m import MGroup
from src.cortical.group.types.z import ZGroup

class SNWRegion(Region):
    def __init__(self, region_index: int, size: int, theta: float):
        region_name = "snw"
        region_id = f"{region_name}_{region_index}"
        omega_group = OmegaGroup(size, theta)
        m_group = MGroup(size, theta)
        z_group = ZGroup(size, theta)
        super().__init__(region_id, [omega_group.group_id, m_group.group_id, z_group.group_id])


    @classmethod
    def create_regions(cls, n: int):
        return [cls(i, n, 0.1) for i in range(n)]