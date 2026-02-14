from __future__ import annotations

import numpy as np

from src.map.input.base import build_region_payloads
from src.map.input.mnist import MnistTiledGlobalMap
from src.map.input.local import FlatLocalInputMap


def main() -> None:
    # Local input map example (single chunk -> single region payload)
    region_id = "R0_0"
    chunk = np.array([[0.1, 0.7], [0.3, 1.0]], dtype=np.float32)

    mapper = FlatLocalInputMap(expected_size=4)
    payload = mapper.map_chunk_to_neurons(region_id=region_id, chunk=chunk)

    print("=== Local Input Map Example ===")
    print("Input chunk:")
    print(chunk)
    print("\nMapped neuron payload:")
    for neuron_id, value in payload.items():
        print(f"{neuron_id}: {value:.2f}")

    # Global input map example (whole image -> multiple region chunks -> payloads)
    image = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    global_map = MnistTiledGlobalMap(region_grid_shape=(2, 2), input_shape=(4, 4), overlap=0)
    local_maps = {
        "R0_0": FlatLocalInputMap(expected_size=4),
        "R0_1": FlatLocalInputMap(expected_size=4),
        "R1_0": FlatLocalInputMap(expected_size=4),
        "R1_1": FlatLocalInputMap(expected_size=4),
    }
    region_payloads = build_region_payloads(global_map=global_map, local_maps=local_maps, sample=image)

    print("\n=== Global Input Map Example ===")
    print("Input image:")
    print(image)
    print("\nPer-region payload sizes:")
    for rid, rpayload in sorted(region_payloads.items()):
        print(f"{rid}: {len(rpayload)} neurons")


if __name__ == "__main__":
    main()
