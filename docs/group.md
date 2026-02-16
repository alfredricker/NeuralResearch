# Groups
`Groups` are what I am calling micro structures such as `M` neurons, `W` neurons, etc. These classes can encode group specific gating mechanisms, weight initializations, decay rates, and what the input streams should be.
For example, the standard `M` group should receive feedforward drive from the omega group of the same region, recurrent drive, and gating drive from `W` neurons. For now I should code something like an `MMW` group (M minus W, meaning no where gating).
We should also define the output stream. The `M` group should ouput feedforward information to the region's `Z` group and feedback information to lower level `omega` groups.

You can also define lateral inhibition for certain groups: How much total activity could there be? How should winners within the group emerge?