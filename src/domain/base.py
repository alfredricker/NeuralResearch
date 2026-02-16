from __future__ import annotations

from typing import Iterable, List, Tuple
from src.settings import MAX_DISCRETE_CARDINALITY
import numpy as np

'''
Domain:
1. Clarify what is the tensor to be mapped to activities (if exists).
2. Normalize it to (0,1)^(n)
3. Clarify the size of the discrete set which is to be mapped to neuron_ids (let this be cardinality k).
3. Surject it to a set of (possibly overlapping) neurons of size k \times m \times n across different regions (optional)
where m is the number of neurons that receive a given item in the set.
4. If no continuous activity tensor exists, then map the discrete terms to have activity a = 1 on their respective neuron_ids.

This class should encapsulate:
- normalization
- surjection to neurons
- cardinality validation
- discrete order validation
'''

class Domain:
    def __init__(
        self,
        activity_tensor_shape: Tuple[int, ...],
        neurons_per_point: int,
        discrete_set_cardinality: int,
        value_range: Tuple[float, float] = (0.0, 1.0),
        discrete_order: Iterable[int] | None = None,
    ):
        self.activity_tensor_shape = activity_tensor_shape
        self.neurons_per_point = neurons_per_point # number of neurons that receive a given item in the set
        self.discrete_set_cardinality = discrete_set_cardinality # number of items in the discrete set
        self.value_range = value_range

        if self.neurons_per_point < 1:
            raise ValueError("neurons_per_point must be >= 1")
        if self.discrete_set_cardinality < 1:
            raise ValueError("discrete_set_cardinality must be >= 1")
        if self.discrete_set_cardinality > MAX_DISCRETE_CARDINALITY:
            raise ValueError(
                "discrete_set_cardinality exceeds MAX_DISCRETE_CARDINALITY "
                f"({self.discrete_set_cardinality} > {MAX_DISCRETE_CARDINALITY})"
            )

        if discrete_order is None:
            self._discrete_order = list(range(self.discrete_set_cardinality))
        else:
            self._discrete_order = list(discrete_order)
        self._validate_discrete_order()

    def _validate_discrete_order(self) -> None:
        if len(self._discrete_order) != self.discrete_set_cardinality:
            raise ValueError(
                "discrete_order length must equal discrete_set_cardinality "
                f"({len(self._discrete_order)} != {self.discrete_set_cardinality})"
            )
        if len(set(self._discrete_order)) != self.discrete_set_cardinality:
            raise ValueError("discrete_order must contain unique indices")
        if set(self._discrete_order) != set(range(self.discrete_set_cardinality)):
            raise ValueError(
                "discrete_order must be a permutation of [0, discrete_set_cardinality)"
            )

    def validate_cardinality(self) -> None:
        activity_size = int(np.prod(self.activity_tensor_shape))
        if activity_size > 0 and activity_size != self.discrete_set_cardinality:
            raise ValueError(
                "activity_tensor_shape cardinality and discrete_set_cardinality mismatch "
                f"({activity_size} != {self.discrete_set_cardinality})"
            )

    def normalize_activity(self, sample: np.ndarray) -> np.ndarray:
        arr = np.asarray(sample, dtype=np.float32)
        if arr.shape != self.activity_tensor_shape:
            raise ValueError(
                f"expected shape {self.activity_tensor_shape}, got {arr.shape}"
            )
        lo, hi = self.value_range
        if hi <= lo:
            raise ValueError("value_range upper bound must be > lower bound")

        min_val = float(arr.min())
        max_val = float(arr.max())
        if min_val == max_val:
            # Degenerate case: all values identical.
            return np.full_like(arr, fill_value=lo, dtype=np.float32)

        scaled = (arr - min_val) / (max_val - min_val)
        return lo + scaled * (hi - lo)

    def discrete_order(self) -> List[int]:
        return list(self._discrete_order)

    def omega_neuron_count(self) -> int:
        return self.discrete_set_cardinality * self.neurons_per_point

