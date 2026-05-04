# Events
Events are computation updates that require a message to be sent to other objects. For example:
* An NMDA spike occurs on a dendritic branch, propagating a depolarization to the soma
* The soma fires an action potential, propagating to all the postsynaptic terminals
* A global reward message is received

## CPU Cache vs GPU Cache
CPU cache automatically evicts cold (inactive) 64-byte cache lines and loads hot ones.
GPU cache doesn't have an analogous built in mechanism. Moving data between RAM and VRAM has a fixed overhead.
Instead of forming multi-megabyte partitions of the data, in a highly dynamic environment such as a biologically plausible GNN, it may make sense to hot load partitions in the kilobyte range based on recent activity.

## Hawkes Process
The assumption behind this memory management process is an object that has received an event is more likely to receive another one in the near future than one that didn't.

## Object Cluster Granularity
One neuron with all its dendritic branches and synapses can form a partition in the kilobyte range. 

## Partitions
A partition is a set of objects loaded into memory.

### Graph Partitioning (e.g. METIS Algorithm)
Partition as to minimize the number of edges crossing boundaries.
**Pros:**
This allows you to theoretically minimize cross partition traffic and it is well studied.
**Cons:**
Have to redefine partitions when connectivity changes. It is an expensive computation.

### Biological Partitioning
You may also partition based on the biological structure, e.g. cortical columns, neural layers, or regions of the artificial brain.


# Clocks
You can run a global clock of a u64, while individual components can track time since last event through u16's.