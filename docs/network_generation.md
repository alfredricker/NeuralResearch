To stitch together regions into a comprehensive network, you first must define all the templates of your regions.
At a fundamental level, the network is just neurons connected to other neurons, the regions are just a useful way of bucketing neurons into types and forming repetitive structure.

Template:
INPUT DOMAINS:
A region template for example would have an id prefix, a local sensory map, and a defined initial number of omega neurons, z neurons, M neurons, etc. You must also define the edge pattern between neurons.
Once you have your input templates, you define the L_S layer, which is how many of each template there is to be. When you input domain data of a given type into the system, it activates the appropriate region types and neurons defined by the maps.

INTERMEDIATE LAYERS:
For intermediate layers, you should define regions prefixed by the layer (this is useful for defining edge patterns).
You could define L1C for example as an intermediate region that connects to both L_S, L2, and a few regions in L1. You must also define the initial structure of the neuron types within the region and the edge patterns between them.