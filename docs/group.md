# Groups
`Groups` are what I am calling micro structures such as `M` neurons, `W` neurons, etc. These classes can encode group specific gating mechanisms, weight initializations, decay rates, and what the input streams should be.
For example, the standard `M` group should receive feedforward drive from the omega group of the same region, recurrent drive, and gating drive from `W` neurons. For now I should code something like an `MMW` group (M minus W, meaning no where gating).
We should also define the output stream. The `M` group should ouput feedforward information to the region's `Z` group and feedback information to lower level `omega` groups.

You can also define lateral inhibition for certain groups: How much total activity could there be? How should winners within the group emerge?

## Defining mappings between groups
When building groups, you define the types of inputs you expect, and the minimimum number of inputs per neuron. Then when building the region or the whole network, you can form edges accordingly.
To define the types of inputs you expect, you need to define the syntax for "source types".
Let's say you have group types 'm', 'z', and 't' in your network.
If you expect group 'm' to have input from 'z' from the same region, then
`source_type='z'` or `source_type='z='`
If you expect group 'm' to have input from 't' from a deeper region (one in a layer with a larger index)
`source_type='t+'`
Or from a different region in the same layer
`source_type='t|'`
From a "shallower region" (layer with a smaller index)
`source_type='t-'`

From external domain (sensory neuron specific)
`source_type='dom_{domain_id}`
This can tell the compiler that these neurons receive information from the domain according to the map defined in the class with the domain id.

## Notes on types of groups
Z groups can apply to any type of region including effector regions. All the effector needs to know is which groups z neurons it receives the data from.