from src.cortical.neuron.edge_topology import DenseTopology, RingTopology, StochasticTopology

# Example neuron ids from two groups
src_ids = [f"m0_{i}" for i in range(8)]     # source group neurons
dst_ids = [f"omega0_{i}" for i in range(12)] # destination group neurons

# 1) Dense group->group projection
dense = DenseTopology()
ff_edges = dense.make_edges(src_ids=src_ids, dst_ids=dst_ids, weight=0.15)

# 2) Stochastic group->group projection
stoch = StochasticTopology(min_fan_in=2, extra_connection_prob=0.1, seed=7)
fb_edges = stoch.make_edges(src_ids=src_ids, dst_ids=dst_ids, weight=0.05)

# 3) Recurrent wiring (same group as src and dst)
recurrent_ids = [f"m0_{i}" for i in range(8)]
ring = RingTopology()
recurrent_edges = ring.make_edges(
    src_ids=recurrent_ids,
    dst_ids=recurrent_ids,
    weight=0.2,
)