## Example of Cortical Dynamics
```
dynamics CorticalDynamics {
    // State variables per node
    state {
        activation: f32 = 0.0;
        input_buffer: f32 = 0.0;
    }
    
    // State variables per synapse
    synapse_state {
        weight: f32 = init_sparse_random(0.0, 0.1);
    }
    
    // Execution phases per tick
    tick {
        // Phase 1: propagate activations through synapses (parallel over synapses)
        parallel for syn in synapses {
            let pre_act = pre(syn).activation;
            post(syn).input_buffer += sigma(pre_act) * syn.weight;
        }
        
        // Phase 2: update activations (parallel over neurons)
        parallel for n in neurons {
            n.activation = (1 - leak) * n.activation + sigma(n.input_buffer);
            n.input_buffer = 0.0;
        }
        
        // Phase 3: learning (parallel over synapses)
        parallel for syn in synapses {
            let pre_act = pre(syn).activation;
            let post_act = post(syn).activation;
            syn.weight += eta * pre_act * post_act * (1 - syn.weight);
        }
    }
}
```

## Example of Predictive Coding Dynamics
```
dynamics PredictiveCoding {
    state {
        activation: f32 = 0.0;
        prediction: f32 = 0.0;
        error: f32 = 0.0;
    }
    
    tick {
        // Top-down predictions
        parallel for syn in feedback_synapses {
            post(syn).prediction += pre(syn).activation * syn.weight;
        }
        
        // Compute errors
        parallel for n in neurons {
            n.error = n.activation - n.prediction;
            n.prediction = 0.0;
        }
        
        // Bottom-up error propagation
        parallel for syn in feedforward_synapses {
            post(syn).activation += pre(syn).error * syn.weight;
        }
        
        // Settle until convergence
        repeat until delta(activation) < epsilon {
            // ... settling dynamics
        }
    }
}
```

## Example of Backprop Dynamics
```
dynamics Backprop {
    state {
        activation: f32 = 0.0;
        gradient: f32 = 0.0;
    }
    
    forward {
        sequential for layer in topological_order {
            parallel for n in layer {
                n.activation = sigma(sum(inputs(n)));
            }
        }
    }
    
    backward {
        sequential for layer in reverse_topological_order {
            parallel for n in layer {
                n.gradient = sum(
                    for syn in outgoing(n): 
                        post(syn).gradient * syn.weight * dsigma(n.activation)
                );
            }
            parallel for syn in layer.synapses {
                syn.weight -= lr * pre(syn).activation * post(syn).gradient;
            }
        }
    }
    
    tick {
        forward;
        loss = compute_loss(output, target);
        output.gradient = dloss(output, target);
        backward;
    }
}
```
