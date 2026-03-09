# Neuron-Astrocyte Network Dynamics & Reinforcement Learning

This repository provides a semi-high level PyTorch implementation and analysis of the neuron-astrocyte network model inspired by the research paper:  
> **[Astrocytes as a mechanism for contextually-guided network dynamics and function](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012186)** (PLOS Computational Biology).

Unlike traditional Artificial Neural Networks (ANNs), this project implements a **tripartite synapse model**, introducing astrocytes as a slow-dynamics variable that modulates synaptic weights to guide contextual switching. The model is tested in a Reinforcement Learning environment (Multi-Armed Bandits) using the REINFORCE algorithm.

---

## Architecture & Mathematical Formulation

The network consists of fast-spiking neurons ($x$) and slow-modulating astrocytes ($z$). The continuous-time Ordinary Differential Equations (ODEs) from the theoretical model were discretized using the **Forward Euler method** to allow for computational modeling.



The discrete-time update rules implemented in the `forward` pass are:

$$x_{t+1} = (1 - \gamma)x_t + \gamma W_t \phi(x_t) + W_{in1} I$$
$$W_{t+1} = (1 - \gamma)W_t + \gamma \left( C \odot \Phi(x_t) + D \psi(z_t) \right)$$
$$z_{t+1} = (1 - \gamma \tau)z_t + \gamma \tau \left( F \odot \psi(z_t) + H \Phi(x_t) + W_{in2} I \right)$$

Where:
* $\gamma$ represents the integration step / learning rate for fast variables.
* $\tau \ll 1$ represents the time-constant for astrocytes (enforcing slow dynamics).
* $\phi$, $\Phi$, $\psi$ are non-linear activation functions (Sigmoid, Outer Product, Tanh).
* $\odot$ denotes element-wise (Hadamard) multiplication.

---

## Engineering Challenges & Optimizations

Translating biological ODEs into efficient PyTorch tensor operations required solving several numerical and architectural challenges:

1.  **Memory Optimization (Avoiding $O(N^2)$ bottlenecks):** Initial implementations using sparse diagonal matrices (`torch.diag(C) @ X`) resulted in massive memory allocations (e.g., over 1GB for a $128 \times 128$ network). This was optimized by mathematically reformulating the matrix multiplication into element-wise operations (`C.unsqueeze(1) * X`), drastically reducing VRAM/RAM consumption to a few megabytes and allowing CPU-only training.
2.  **Truncated Backpropagation Through Time (TBPTT):** To train the dynamic system using Policy Gradients without exhausting computational graphs, the state variables ($x, W, z$) are maintained across steps but `.detach()`ed at fixed intervals (e.g., `bptt_steps=20`). This prevents exploding gradients while preserving the long-term dependency required by the slow astrocytic variables.
3.  **Numerical Stability:** Careful selection of the $\gamma$ step was required to prevent the Euler approximation from diverging (truncation error accumulation) while bridging the time-scale separation between neurons (fast) and astrocytes (slow).

---

## Reinforcement Learning Analysis (Multi-Armed Bandits)

The model was tested in a 6-armed Bernoulli Bandit environment using a custom Policy Gradient (REINFORCE) loop with an Exponential Moving Average (EMA) baseline.



### Behavioral Observations: The Local Minimum Problem
During testing in **stationary** environments (where reward probabilities remain constant), the network sometimes exhibits premature convergence, locking into a sub-optimal arm. This manifests as a linear growth in cumulative regret. 

**Why does this happen?**
This behavior actually highlights the core nature of the astrocytic model. In a stationary environment, fast neurons quickly adapt to the first "good enough" reward, halting exploration. Because the context never changes, the slow astrocytes ($z$) stabilize and fail to trigger a contextual switch. 

The true power of this tripartite architecture, as described in the PLOS paper, emerges in **non-stationary (fluctuating) environments**. When reward distributions suddenly shift, the neurons remain trapped in their local minimum. It is the slow accumulation of error signals by the astrocytes that eventually provides a massive modulatory signal, effectively "kicking" the neurons out of the local minimum and forcing the network to explore and adapt to the new context.

---

## Getting Started

### Prerequisites
* Python 3.10+
* PyTorch
* NumPy
* Matplotlib

### Running the model
To run the Multi-Armed Bandit simulation and plot the cumulative regret:
```bash
python impl.py