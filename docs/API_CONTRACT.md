# API Contract — Non-Orientable Neural Networks

**Version**: v0.1-topology-locked
**Status**: STABLE
**Last Updated**: 2026-01-05

This document defines the **public API contract** for the non-orientable neural network implementation. Any changes to these interfaces require updating this document, tests, and the changelog.

---

## Non-Negotiable Invariants

These properties are **mathematically enforced** and must never be violated:

### 1. Parity Operator Involution
```python
S @ S == I  # Verified to tolerance 1e-6
```
Eigenvalues ∈ {+1, -1} exactly.

### 2. Projector Properties
```python
P_plus @ P_plus == P_plus      # Idempotent
P_minus @ P_minus == P_minus   # Idempotent
P_plus @ P_minus == 0          # Orthogonal
P_plus + P_minus == I          # Partition
```

### 3. Weight Commutation/Anticommutation
```python
W_comm @ S == S @ W_comm       # Commutant
W_flip @ S == -S @ W_flip      # Anticommutant
```

### 4. Loss Function Invariance
Quotient loss must satisfy:
```python
loss(y, ŷ) == loss(-y, ŷ)      # Sign invariance
loss(y, ŷ) == loss(y, -ŷ)
loss(y, ŷ) == loss(-y, -ŷ)
```

### 5. Benchmark Metrics
All benchmark runs must produce:
- `overall_mse`: Mean squared error across entire sequence
- `within_regime_mse`: MSE outside transition windows
- `transition_mse`: MSE within ±20 steps of regime switches
- `params`: Total parameter count

### 6. Test Markers
- `@pytest.mark.slow`: Training/integration tests (>5 seconds)
- Fast tests: Unit tests, property checks (<1 second each)

---

## Stable Public API

### Module: `src/parity.py`

#### Classes

**`ParityOperator(dim: int, even_dim: int = None)`**
- Constructs ℤ₂ parity operator S
- **Guarantee**: S² = I
- **Attributes**: `S` (tensor), `dim`, `even_dim`, `odd_dim`
- **Methods**: `__call__(h)`, `to(device)`

**`ParityProjectors(parity_op: ParityOperator)`**
- Constructs P₊, P₋ projectors
- **Guarantee**: Projector properties (idempotent, orthogonal, partition)
- **Methods**:
  - `project_plus(h)`, `project_minus(h)`
  - `parity_energy(h) -> Tensor`: Returns α₋ ∈ [0, 1]
  - `to(device)`

#### Functions

**`verify_involution(S: Tensor, tol: float = 1e-6) -> bool`**
**`verify_eigenvalues(S: Tensor, tol: float = 1e-6) -> bool`**
**`verify_projector_properties(P_plus, P_minus, tol: float = 1e-6) -> dict`**
**`verify_commutation(W: Tensor, S: Tensor, tol: float = 1e-6) -> bool`**
**`verify_anticommutation(W: Tensor, S: Tensor, tol: float = 1e-6) -> bool`**
**`construct_commutant_weight(A_plus, A_minus, parity_op) -> Tensor`**
**`construct_anticommutant_weight(B_plus_minus, B_minus_plus, parity_op) -> Tensor`**

---

### Module: `src/data.py`

#### Classes

**`AntipodalRegimeSwitcher(latent_dim, obs_dim, p_switch=0.05, ...)`**
- Generates antipodal regime-switching time series
- **Guarantee**: Rank(C) < latent_dim (partial observability)
- **Methods**:
  - `generate_sequence(T) -> (obs, latents, regimes)`
  - `verify_antipodal_symmetry(z) -> float`
  - `verify_observation_rank() -> int`
  - `estimate_switch_rate(T) -> float`

#### Functions

**`create_train_test_split(generator, train_length, test_length, seed) -> (train_data, test_data)`**
**`find_regime_switches(regimes, window=20) -> (switch_times, transition_mask)`**

---

### Module: `src/models.py`

#### Classes

**`Z2EquivariantRNN(input_dim, hidden_dim, output_dim, even_dim=None)`**
- Commutant-only model (no seam coupling)
- **Guarantee**: Weight matrix is block-diagonal (no parity mixing)
- **Methods**:
  - `forward(x, h=None) -> (outputs, h_final)`
  - `step(x, h) -> h_next`
  - `get_weight_matrices() -> (W_comm, W_flip)`

**`SeamGatedRNN(input_dim, hidden_dim, output_dim, gate_type, ...)`**
- Full seam-gated model
- **gate_type**: 'fixed', 'learned', 'kstar'
- **Guarantee**: Gate output ∈ [0, 1]
- **Methods**:
  - `forward(x, h=None) -> (outputs, h_final, gates)`
  - `step(x, h) -> (h_next, g)`
  - `compute_gate(h) -> g`

**`GRUBaseline(input_dim, hidden_dim, output_dim)`**
- Standard GRU for comparison
- **Methods**: `forward(x, h=None) -> (outputs, h_final, gates)`

---

### Module: `src/losses.py`

#### Functions

**`quotient_loss(y_true, y_pred) -> Tensor`**
- **Guarantee**: Sign-invariant (projective)
- Returns mean min-distance loss

**`rank1_projector_loss(y_true, y_pred) -> Tensor`**
- **Guarantee**: Sign-invariant via outer products
- Returns Frobenius norm of projector difference

**`verify_quotient_invariance(loss_fn, y, y_pred, tol=1e-6) -> bool`**
**`verify_rank1_equivalence(y, y_pred, tol=1e-5) -> bool`**

#### Classes

**`QuotientLoss(nn.Module)`**
**`Rank1ProjectorLoss(nn.Module)`**

---

### Module: `src/baselines.py`

#### Classes

**`AR1Model(obs_dim)`**
- **Methods**:
  - `fit(observations)`
  - `predict(x) -> predictions`
  - `verify_residuals() -> (is_finite, is_spd)`

**`IMMFilter(obs_dim, p_switch=0.05)`**
- Interacting Multiple Model filter
- **Methods**:
  - `fit(observations, regimes=None)`
  - `predict(x) -> (prediction, mode_probs)`
  - `verify_mode_probs() -> bool`

#### Functions

**`compute_transition_error_spike(errors, regimes, window=5) -> (transition_error, stable_error)`**

---

## Internal / Unstable

The following are **implementation details** and may change:
- Private methods (prefixed with `_`)
- Internal weight initialization schemes
- Specific optimizer configurations in benchmarks
- Figure styling details

---

## Deprecation Policy

1. **Breaking changes**: Must increment version tag (v0.1 → v0.2)
2. **Deprecated functions**: Marked with warning for ≥1 release before removal
3. **Test failures**: Any API change that breaks existing tests requires explicit justification

---

## Enforcement

This contract is enforced by:
- `tests/test_02_parity_structure.py` (invariants)
- `tests/test_03_forward_dynamics.py` (model guarantees)
- `tests/test_05_loss_functions.py` (loss properties)
- CI pipeline (all tests must pass)

**Last Verified**: 2026-01-05, commit 7baf091
