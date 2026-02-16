How should regions be structured?
We should start with a template class that can be inherited from for defining the specific regions of the whole network.

In order to stitch regions together at network build time, the F_omega and F_z sets need to be passed edge patterns.

The SensoryRegion class takes a LocalInputMap as an argument (defaults to FlatLocalInputMap).
The EffectorRegion takes a LocalOutputMap as an argument.

When building the network, you need to define (1 or more) GlobalInputMap to define how the input data gets mapped to the sensory neurons, as well as (at least one) GlobalOutputMap to define what the output of multiple regions gets translated to. For a classifier, it makes sense to have this be a voting mechanism.

The way the current output map code works is that we create n effector neurons per output region with IDs like ``{region_id}:z_{k}``. We then assign the neurons ids based on these n objects. We parse the activation values into a dictionary, taking the argmax as the one that the network predicts. We then vote on the regions.