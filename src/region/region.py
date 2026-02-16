from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.neuron.edge import Edge
from src.neuron.neuron import BaseNeuron


class Region:
    def __init__(self, region_id: str):
        self.region_id = region_id
        self.neurons = {}
        self.edges = []
    
