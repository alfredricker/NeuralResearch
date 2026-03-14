# Purpose
The purpose of a network is to take in data, process it, and output data. Therefore I adopt the simple general framework where *Nodes* are pieces of data and *Edges* transform the data either symmetrically or directionally.

# Graph Declaration
The final object of interest is our complete neural network. This is the graph that we can apply learning rules to, feed streams of information, train, etc. Let's start by declaring a disconnected graph of 50 `nodes`. 

```stn
graph {
    nodes(50);
}
```

This graph simply generates 50 empty node objects that could be loaded into memory.

### Declaring edges

Edges connect nodes. The arrow `->` creates directed connections from left to right.
```stn
graph {
    graph_nodes = nodes(50);
    graph_nodes -> graph_nodes all;
}
```

This creates 50 nodes where every node connects to every other node (2500 edges).

The topology `all` specifies the connection pattern. Other patterns:
```
nodes -> nodes: sparse(0.1);   // 10% of possible edges, random -- no identity connections
nodes -> nodes: identity;      // node i connects to node i only
nodes -> nodes: ring(1);       // node i connects to node (i+1) mod n
nodes -> nodes: none;          // no connections
```