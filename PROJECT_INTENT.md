# Non-Orientable Neural Networks (ℤ₂ Seam-Gated) — Project Intent

## Purpose

This project develops and rigorously evaluates neural network architectures whose hidden states are structured to behave like a **non-orientable quotient space**—specifically the projective geometry that arises when identifying antipodal points (x ∼ -x). We operationalize this through a **ℤ₂ parity decomposition** of representations and a **seam gate** that activates "chart transitions" when the data demands it.

The point is not geometric aesthetics. The point is a measurable, repeatable win on problems where **antipodal dynamics are real** and standard architectures are structurally mismatched.

---

## Core Idea in One Sentence

We enforce a **parity split** (even/odd channels) and add a **learnable seam-crossing operator** whose activation is controlled by a principled **phase boundary** (including the constant k* ≈ 0.721), so the network can navigate **quotient geometry** rather than duplicating capacity for sign-flipped regimes.

---

## Why This Matters

Many real signals contain "same underlying mechanism, opposite orientation" patterns:

* regime switching where the same dynamics apply but **sign flips**
* control systems with heating/cooling or forward/reverse mode symmetry
* sensors where calibration or polarity creates an ambiguity (x) vs (-x)
* inverse problems where the physical measurement destroys sign (projective ambiguity)

Vanilla networks typically treat sign-flipped regimes as unrelated. This causes:

1. **capacity duplication** (learning two copies of the same rule)
2. **error spikes** at regime boundaries (relearning after switches)
3. poor transfer across different switch rates and environments

Our architecture is designed so the symmetry is **built into the hypothesis class**.

---

## What We Are Building

### 1) ℤ₂-Equivariant Representation Space

We split hidden state h ∈ ℝⁿ into parity channels:

* h₊ = P₊ h (even)
* h₋ = P₋ h (odd)

where P± = ½(I ± S) and S² = I is the parity operator.

### 2) Two Kinds of Learnable Maps

**(A) Parity-preserving map (commutant):**
W_comm S = S W_comm
This preserves charts / parity subspaces.

**(B) Parity-swapping map (anticommutant):**
W_flip S = -S W_flip
This swaps even/odd channels and represents seam crossing / chart transitions.

### 3) Seam Gate (Adaptive Chart Transition)

We update hidden state using a mixture:

h_{t+1} = σ(W_comm u_t + g(h_t) W_flip S u_t + b), where u_t = [h_t, x_t]

The seam gate g(h_t) ∈ [0,1] decides when to activate parity swapping.

### 4) Phase Control via Parity Energy and k*

Define "odd energy":

α₋(h) = ||P₋ h||² / ||h||² ∈ [0,1]

The k*-gated version uses:

g(h) = sigmoid((α₋(h) - k*) / τ)

Interpretation: the model stays chart-local when structure is stable, and activates seam transitions when parity evidence rises past a critical boundary.

---

## What We Are Trying to Achieve (Measurable Objectives)

### Primary Objective (Core Claim)

On antipodal regime-switching time series, the full model should:

1. **reduce transition error** (MSE near switch times) vs standard baselines
2. **reduce transient spikes** at boundaries vs GRU/LSTM/Transformer baselines
3. show **parameter efficiency** (better transition performance at equal or fewer parameters)

### Mechanism Objective (Not Just "It Works")

We should observe:

* gate g(t) **peaks near true switch times**
* α₋(t) **rises toward the phase boundary** near switches
* ablations prove each component is load-bearing:
  * commutant only < fixed gate < learned gate < k*-gated seam

### Secondary Objective (Generalization)

Train on one switch probability and test on higher switch probability; the non-orientable model should degrade **less** than vanilla networks.

---

## Benchmark Plan

### Synthetic Canonical Benchmark (Ground Truth)

Latent dynamics:

* Regime A: z_{t+1} = A z_t + ε_t
* Regime B: z_{t+1} = -A z_t + ε_t

Observed signal:

x_t = C z_t + η_t

with Markov switching of regime. This makes the antipodal geometry mathematically necessary.

### Baselines

* GRU / LSTM (standard recurrent)
* ℤ₂ commutant-only model (no seam coupling)
* Seam coupling with fixed gate (g=0.5)
* Seam coupling with learned gate (no k*)
* Full k*-gated seam model
* Classical baseline:
  * non-oracle switching AR(1) IMM fitted from training data

### Metrics

* overall one-step MSE
* within-regime MSE
* transition-window MSE (±20 around switches)
* switch-aligned error curves
* gate and α₋ switch-aligned curves

---

## Deliverables

1. **Reference implementation** of all model variants and baselines
2. **Reproducible experiment harness**
   * multi-seed aggregation
   * CSV outputs and LaTeX table export
   * saved plots for paper figures
3. **Test suite** enforcing:
   * parity operator correctness (S²=I)
   * commutation/anticommutation properties of learned maps
   * gate monotonicity and k* behavior
   * no NaNs / stability
   * benchmark ordering and ablation necessity

---

## Success Criteria (What "Done" Looks Like)

We consider this project successful when:

* the full model beats standard neural baselines on **transition error** reliably across seeds
* the ablation ladder is correct (each component improves what it claims to improve)
* the gate/α₋ plots align with switches and are stable across runs
* classical non-oracle switching AR baseline is competitive but does not eliminate the advantage
* results are packaged into a clean repo with:
  * tests
  * scripts
  * figures
  * a paper-ready table and narrative

---

## One-Line Thesis

We are turning **non-orientability** from a metaphor into an **architectural constraint + mechanism** that measurably improves learning in systems with **antipodal regime structure**, using seam gating governed by a principled phase boundary.
