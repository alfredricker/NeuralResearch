How should regions be structured?
We should start with a template class that can be inherited from for defining the specific regions of the whole network.

In order to stitch regions together at network build time, the F_omega and F_z sets need to be passed edge patterns.

The network structure:
1. Smallest structure is neuron. This requires at least two `Edges` connecting to it. An input map can from a single datapoint to a single neuron or an output map from a single neuron to an element of a tensor or any decode mode can be thought of as edges.
2. We want to build the network from small structures (Groups) stitched together by edge patterns.
3. Building structures should be efficient and reusable. If we build a certain structure, it should be able to be placed in different contexts.
For example, say you build a recurrent structure of `M` neurons where M1 -> M2 -> M3 -> M1. This should be abstracted to its essential structure so that it can be reused in different regions and connect to different structures.