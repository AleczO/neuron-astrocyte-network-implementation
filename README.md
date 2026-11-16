# Neuron-Astrocyte Network Dynamics & Reinforcement Learning

A PyTorch reimplementation and analysis of the neuron-astrocyte network model from:

> **[Astrocytes as a mechanism for contextually-guided network dynamics and function](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012186)**
> Gong, Pasqualetti, Papouin & Ching, *PLOS Computational Biology* 20(5): e1012186 (2024).

Unlike a standard ANN, this model implements a **tripartite synapse**: astrocytes enter as a slow-dynamics variable that modulates synaptic weights. The network is evaluated in a reinforcement learning setting (multi-armed bandits) and trained with REINFORCE.

This is a reimplementation written from the equations in the paper rather than a fork of the authors' code. Where it departs from the reference implementation, the differences are listed under *Known limitations*.

---

## Model and discretisation

The network consists of fast neurons ($x$), fast synapses ($W$) and slow astrocytes ($z$). The paper's continuous-time ODEs

$$\tau\dot{x} = -x + W\phi(x), \qquad \tau\dot{W} = -W + C\Phi(x) + D\psi(z), \qquad \dot{z} = -z + F\psi(z) + H\Phi(x)$$

are first **rescaled in time** ($s = t/\tau$). This moves $\tau$ onto the astrocytic equation and leaves the neuronal and synaptic equations with unit relaxation rate, while preserving the time-scale separation between them. The rescaled system is then discretised with the **forward Euler method** (step $\gamma$):

$$x_{t+1} = (1-\gamma)x_t + \gamma\left(W_t\phi(x_t) + W_{in1}I\right)$$

$$W_{t+1} = (1-\gamma)W_t + \gamma\left(C \odot \Phi(x_t) + D\psi(z_t)\right)$$

$$z_{t+1} = (1-\gamma\tau)z_t + \gamma\tau\left(F \odot \psi(z_t) + H\Phi(x_t) + W_{in2}I\right)$$

where:

* $\gamma$ is the integration step.
* $\tau \ll 1$ is the astrocytic time constant, enforcing slow astrocytic dynamics.
* $\phi$ = sigmoid, $\psi$ = tanh, and $\Phi(x) = \phi(x)\phi(x)^{T}$ flattened to a vector of length $n^2$.
* $\odot$ denotes an element-wise product (see below).

The full derivation, including the time rescaling and the tensor shapes, is in `Calculations.ipynb`.

---

## Engineering notes

**Memory: avoiding an explicit diagonal matrix.** $\Phi(x)$ has $n^2$ entries, so a general $C$ would be $n^2 \times n^2$. Both this and the reference implementation restrict $C$ to a diagonal, but the reference materialises it as `torch.diag(C) @ X`. For $n = 128$ that allocates a $16384 \times 16384$ float32 matrix, i.e. 1 GiB per step. Since `diag(C) @ X` is exactly the element-wise product `C * X`, replacing it with `C.unsqueeze(1) * X` reduces the cost to a few MB and makes CPU-only training feasible.

**Truncated backpropagation through time.** The reference implementation backpropagates and detaches at every step. Here the state $(x, W, z)$ is carried across steps and detached every `bptt_steps=20`, which keeps a longer gradient path through the slow astrocytic variable without exhausting the computation graph.

**Numerical stability.** $\gamma$ has to be small enough that the forward Euler approximation does not diverge, while still bridging the separation between fast neuronal and slow astrocytic time scales.

---

## Reinforcement learning setup

The model is evaluated on a 6-armed Bernoulli bandit, trained with a custom REINFORCE loop using a **running-mean reward baseline**. The evaluation metric is cumulative regret:

$$R_T = \sum_{t=1}^{T}\left(\max_{a_i} \mu_i - \mathbb{E}[r_t]\right)$$

A flattening regret curve indicates the optimal arm has been found; linear growth indicates the agent is stuck on a sub-optimal arm.

---

## Status and known limitations

* **Only the stationary environment is implemented.** `SBDataset` draws the arm probabilities once, so the reward distribution never changes and the external cue $I$ is held constant. The paper's non-stationary conditions (flip-flop switching and smooth periodic drift) are not implemented here.

* **Premature convergence in the stationary case.** The network sometimes locks onto a sub-optimal arm, which shows up as linearly growing cumulative regret.

* **The astrocytic mechanism is not tested here.** The paper argues that the slow astrocytic variable is what supports adaptation when the context shifts, and that the time-scale separation is necessary for learning across changing contexts. That claim cannot be evaluated in a stationary environment, and nothing in this repository tests it. Testing it would require implementing the non-stationary bandit together with a varying cue, and comparing against an ablation in which $z$ is frozen.

* **Deviations from the reference implementation**, which may affect the behaviour above:
  * $F$ is restricted to a diagonal here; the reference uses a full $m \times m$ matrix.
  * The synaptic update uses $\Phi(x_t)$, whereas the reference uses the already-updated $\Phi(x_{t+1})$.
  * No temperature is applied to the policy logits; the reference divides them by 2, which makes the sampled policy more exploratory.

---

## Getting started

### Prerequisites

* Python 3.10+
* PyTorch
* NumPy
* Matplotlib

### Running the model

To run the multi-armed bandit simulation and plot cumulative regret:

```bash
python impl.py
```