from __future__ import annotations

from typing import Iterable, List, Tuple
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

MAX_DISCRETE_CARDINALITY = 10_000

class Domain:
    def __init__(
        self,
        activity_tensor_shape: Tuple[int, ...],
        discrete_set_cardinality: int,
        value_range: Tuple[float, float] = (0.0, 1.0),
        discrete_order: Iterable[int] | None = None,
    ):
        self.activity_tensor_shape = activity_tensor_shape
        self.discrete_set_cardinality = discrete_set_cardinality # number of items in the discrete set
        self.value_range = value_range

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

