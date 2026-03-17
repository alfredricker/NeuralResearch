# Experimental Neural Networks Project
## Goal
My goal with this project is to build a framework in Rust that is capable of building novel neural networks with a well written, effecient, and elegant API. This framework should support creation of more traditional neural networks, as well as the dynamical constructs found in the `tex/` directory.

## Coding Practices
Code written for this project should be reusable and clear. For example, the Cortical Column neural network is built in a hierarchical organization of `Node -> Structure -> Region -> Network`. We should have a clean api for building smaller components that could be glued together by there in and out ports.

As for more traditional neural networks we use the Burn cargo crate. Use the google-ai-mode-skill to inform yourself how to use functions and objects from this crate (or the locally saved Burn code, but the google-ai-mode-skill will likely save tokens).

# The Cortical Column Network Model Overview
If you want to read more in depth notes see `tex/Cortical.tex`

## 1. Core Principles
* **State:** Scalar activation $\alpha_i(t) \in \mathbb{R}$, bounded in $(-1,1)$. No dense embedding vectors.
* **Representation:** Distributed sparse population coding ($|A| \approx k \ll N$).
* **Similarity:** Set overlap of active subsets, not dot products.

## 2. Neuron Dynamics
* **Readout/Saturation:** $\sigma(x) = \frac{x}{|x|+1}$
* **Synaptic Input:** $f_h(t) = \sum_{p \in P(h)} \sigma(\alpha_p(t)) \cdot w(p, h)$
* **Update Rule (with decay $\lambda$):** $\alpha_h(t+1) = (1-\lambda)\alpha_h(t) + \sigma(f_h(t))$

## 3. Regional Sub-Populations
* **$F_\omega$ (Feed-in) & $F_z$ (Feed-out):** Inter-region communication boundaries.
* **$W$ (Where):** Tracks location/context.
    * **$W_\mathbb{T}$:** Hardwired cyclic grid for path integration (uses Chinese Remainder Theorem).
    * **$W_\mathbb{M}$:** Learned context component.
* **$M$ (Model):** Concept representations. Activation requires coincidence of sensory feed-forward ($F$) and location gate ($W$), heavily modulated by lateral inhibition to enforce sparsity:
    $$f_i(t) = f_i^F(t) \cdot g(f_i^W(t)) + f_i^R(t) - \kappa \cdot \sigma(\alpha_i(t)) \cdot E_M(t)$$

## 4. Learning Rules
* **Hebbian Update:** $\Delta w(h_i, h_j) = \eta \cdot \sigma(\alpha_i(t)) \cdot \sigma(\alpha_j(t))$
* **Weight Decay:** $w(t+1) = (1 - \mu) \cdot w(t) + \Delta w(t)$ (where $\mu \ll \eta$)
* **Enzyme Modulation:** $\eta_{\text{eff}} = \eta \cdot \nu(t)$ (scales learning via global error/novelty signal)

## 5. Network Topology
* **Hierarchy ($\mathcal{L}_0 \dots \mathcal{L}_D$):** Directed processing levels.
* **Convergence Axiom:** Association regions ($L_{k>0}$) must integrate feed-out from $\ge 2$ lower-level regions.
* **Lateral Connections:** Modulatory voting within the same level to resolve ambiguity.
* **Top-Down Feedback:** Multiplicative and depolarizing. Expected lower-level features gain a competitive boost (enabling predictive coding).