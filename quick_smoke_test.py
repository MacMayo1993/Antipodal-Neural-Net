"""
Quick smoke test to verify core implementation before full benchmark.
Tests basic functionality of all models on small data.
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

print("=" * 60)
print("QUICK SMOKE TEST")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/6] Testing imports...")
try:
    from src.baselines import AR1Model, IMMFilter
    from src.data import AntipodalRegimeSwitcher
    from src.losses import quotient_loss
    from src.models import GRUBaseline, SeamGatedRNN, Z2EquivariantRNN
    from src.parity import ParityOperator, ParityProjectors
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Parity structure
print("\n[2/6] Testing ℤ₂ parity structure...")
try:
    parity_op = ParityOperator(10, 5)
    S = parity_op.S

    # Verify S² = I
    S_squared = S @ S
    I = torch.eye(10)
    assert torch.allclose(S_squared, I, atol=1e-6), "S² ≠ I"

    # Verify projectors
    projectors = ParityProjectors(parity_op)
    P_plus = projectors.P_plus
    P_minus = projectors.P_minus

    assert torch.allclose(P_plus @ P_plus, P_plus, atol=1e-6), "P₊ not idempotent"
    assert torch.allclose(P_minus @ P_minus, P_minus, atol=1e-6), "P₋ not idempotent"
    assert torch.allclose(P_plus + P_minus, I, atol=1e-6), "P₊ + P₋ ≠ I"

    print("  ✓ Parity operator involution verified")
    print("  ✓ Projector properties verified")
except Exception as e:
    print(f"  ✗ Parity test failed: {e}")
    sys.exit(1)

# Test 3: Data generation
print("\n[3/6] Testing data generation...")
try:
    torch.manual_seed(42)
    np.random.seed(42)

    gen = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, p_switch=0.05, seed=42)
    obs, latents, regimes = gen.generate_sequence(T=100)

    assert obs.shape == (100, 4), f"Obs shape wrong: {obs.shape}"
    assert latents.shape == (100, 8), f"Latent shape wrong: {latents.shape}"
    assert regimes.shape == (100,), f"Regime shape wrong: {regimes.shape}"
    assert torch.all(torch.isfinite(obs)), "NaN/Inf in observations"

    # Verify rank deficiency
    rank = gen.verify_observation_rank()
    assert rank < 8, f"Observation matrix not rank deficient: rank={rank}"

    print(f"  ✓ Generated sequence T=100, obs_dim=4, latent_dim=8")
    print(f"  ✓ Observation rank = {rank} < 8 (partial observability)")
except Exception as e:
    print(f"  ✗ Data generation failed: {e}")
    sys.exit(1)

# Test 4: Model forward passes
print("\n[4/6] Testing model forward passes...")
models_to_test = [
    ('GRU', GRUBaseline(4, 8, 4)),
    ('Z2_equi', Z2EquivariantRNN(4, 8, 4)),
    ('Z2_kstar', SeamGatedRNN(4, 8, 4, gate_type='kstar'))
]

try:
    x = torch.randn(2, 10, 4)  # batch=2, seq=10, dim=4

    for name, model in models_to_test:
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'gru'):
                y, h, g = model(x)
            elif hasattr(model, 'gate_type'):
                y, h, g = model(x)
            else:
                y, h = model(x)
                g = None

        assert y.shape == (2, 10, 4), f"{name}: output shape wrong"
        assert torch.all(torch.isfinite(y)), f"{name}: NaN/Inf in output"

        if g is not None:
            assert torch.all((g >= 0) & (g <= 1)), f"{name}: gate outside [0,1]"

        print(f"  ✓ {name}: forward pass OK")
except Exception as e:
    print(f"  ✗ Model test failed: {e}")
    sys.exit(1)

# Test 5: Quick training loop
print("\n[5/6] Testing training stability (10 steps)...")
try:
    torch.manual_seed(42)
    model = SeamGatedRNN(4, 8, 4, gate_type='kstar')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X = obs[:-1].unsqueeze(0)
    Y = obs[1:].unsqueeze(0)

    for step in range(10):
        optimizer.zero_grad()
        y_pred, _, _ = model(X)
        loss = torch.nn.functional.mse_loss(y_pred, Y)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"Non-finite loss at step {step}"

    print(f"  ✓ Training 10 steps: final loss = {loss.item():.4f}")
except Exception as e:
    print(f"  ✗ Training test failed: {e}")
    sys.exit(1)

# Test 6: Loss functions
print("\n[6/6] Testing loss functions...")
try:
    y = torch.randn(5, 4)
    y_pred = torch.randn(5, 4)

    # Quotient loss
    L = quotient_loss(y, y_pred)
    L_flip_y = quotient_loss(-y, y_pred)
    L_flip_pred = quotient_loss(y, -y_pred)

    assert torch.isfinite(L), "Quotient loss not finite"
    assert torch.abs(L - L_flip_y) < 1e-5, "Quotient loss not sign-invariant (y)"
    assert torch.abs(L - L_flip_pred) < 1e-5, "Quotient loss not sign-invariant (ŷ)"

    print("  ✓ Quotient loss sign-invariance verified")
except Exception as e:
    print(f"  ✗ Loss test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL SMOKE TESTS PASSED")
print("=" * 60)
print("\nReady to run full benchmark.")
