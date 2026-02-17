from src.cortical.group.types.m import MGroup
from src.cortical.group.types.mmw import MMWGroup
from src.cortical.group.types.omega import OmegaGroup
from src.cortical.group.types.w import WGroup
from src.cortical.group.types.z import ZGroup
from src.cortical.group.types.zmw import ZMWGroup

__all__ = [
    "MGroup",
    "MMWGroup",
    "OmegaGroup",
    "WGroup",
    "ZGroup",
    "ZMWGroup",
]

'''
M:
- ff_from_omega: omega=
- gating_from_w: w=
- fb_from_z ?: z=

W:
- ff_from_omega: omega=

Omega:
- ff_from_z: z-
- fb_from_m: m+

Z:
- ff_from_m: m=
- ff_from_w: w=

ZMW:
- ff_from_m: m=

MMW:
- ff_from_omega: omega=

'''