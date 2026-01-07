"""
Test suite for seed utilities and RNG reproducibility.

Tests that the unified seed utilities ensure reproducibility across
CPU/GPU and different random operations.
"""

import pytest
import torch
import numpy as np
import random

from src.seed import set_seed, create_generator, get_rng_state, set_rng_state
from src.data import AntipodalRegimeSwitcher


class TestSetSeed:
    """Test set_seed function"""

    def test_set_seed_torch(self):
        """set_seed should make torch operations reproducible"""
        set_seed(42)
        x1 = torch.randn(10)

        set_seed(42)
        x2 = torch.randn(10)

        assert torch.equal(x1, x2), "torch.randn not reproducible"

    def test_set_seed_numpy(self):
        """set_seed should make numpy operations reproducible"""
        set_seed(42)
        x1 = np.random.randn(10)

        set_seed(42)
        x2 = np.random.randn(10)

        assert np.allclose(x1, x2), "numpy.random not reproducible"

    def test_set_seed_python(self):
        """set_seed should make python random reproducible"""
        set_seed(42)
        x1 = [random.random() for _ in range(10)]

        set_seed(42)
        x2 = [random.random() for _ in range(10)]

        assert x1 == x2, "random.random not reproducible"

    def test_set_seed_all_together(self):
        """set_seed should make all RNG sources reproducible together"""
        set_seed(42)
        torch_val1 = torch.randn(1).item()
        numpy_val1 = np.random.randn()
        python_val1 = random.random()

        set_seed(42)
        torch_val2 = torch.randn(1).item()
        numpy_val2 = np.random.randn()
        python_val2 = random.random()

        assert torch_val1 == torch_val2, "torch not reproducible"
        assert numpy_val1 == numpy_val2, "numpy not reproducible"
        assert python_val1 == python_val2, "python not reproducible"

    def test_different_seeds_produce_different_values(self):
        """Different seeds should produce different random values"""
        set_seed(42)
        x1 = torch.randn(10)

        set_seed(123)
        x2 = torch.randn(10)

        assert not torch.equal(x1, x2), "Different seeds produced same values"


class TestCreateGenerator:
    """Test create_generator function"""

    def test_generator_with_seed(self):
        """Generator with seed should be reproducible"""
        gen1 = create_generator(42)
        x1 = torch.randn(10, generator=gen1)

        gen2 = create_generator(42)
        x2 = torch.randn(10, generator=gen2)

        assert torch.equal(x1, x2), "Generator not reproducible"

    def test_generator_independent_of_global_state(self):
        """Generator should be independent of global RNG state"""
        # Set global seed
        set_seed(999)

        # Create generator with different seed
        gen = create_generator(42)
        x1 = torch.randn(10, generator=gen)

        # Change global seed
        set_seed(777)

        # Create another generator with same seed as before
        gen2 = create_generator(42)
        x2 = torch.randn(10, generator=gen2)

        # Should still be equal
        assert torch.equal(x1, x2), "Generator affected by global state"

    def test_generator_device(self):
        """Generator should respect device parameter"""
        gen_cpu = create_generator(42, device="cpu")
        assert gen_cpu.device.type == "cpu"

        if torch.cuda.is_available():
            gen_cuda = create_generator(42, device="cuda")
            assert gen_cuda.device.type == "cuda"


class TestRNGState:
    """Test RNG state capture and restore"""

    def test_get_and_set_rng_state(self):
        """Should be able to capture and restore RNG state"""
        set_seed(42)

        # Generate some random values
        _ = torch.randn(5)
        _ = np.random.randn(5)
        _ = random.random()

        # Capture state
        state = get_rng_state()

        # Generate more values
        x1_torch = torch.randn(10)
        x1_numpy = np.random.randn(10)
        x1_python = [random.random() for _ in range(10)]

        # Restore state
        set_rng_state(state)

        # Generate values again - should match
        x2_torch = torch.randn(10)
        x2_numpy = np.random.randn(10)
        x2_python = [random.random() for _ in range(10)]

        assert torch.equal(x1_torch, x2_torch), "torch state not restored"
        assert np.allclose(x1_numpy, x2_numpy), "numpy state not restored"
        assert x1_python == x2_python, "python state not restored"


class TestDataGeneratorReproducibility:
    """Test that data generator uses RNG correctly"""

    def test_generator_reproducible_with_seed(self):
        """Data generator should be reproducible with seed parameter"""
        gen1 = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, seed=42)
        obs1, _, reg1 = gen1.generate_sequence(T=100)

        gen2 = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, seed=42)
        obs2, _, reg2 = gen2.generate_sequence(T=100)

        assert torch.allclose(obs1, obs2, atol=1e-6), "Observations not reproducible"
        assert torch.equal(reg1, reg2), "Regimes not reproducible"

    def test_generator_with_torch_generator(self):
        """Data generator should work with torch.Generator"""
        gen_data = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, seed=42)

        gen1 = create_generator(99)
        obs1, _, reg1 = gen_data.generate_sequence(T=100, generator=gen1)

        gen2 = create_generator(99)
        obs2, _, reg2 = gen_data.generate_sequence(T=100, generator=gen2)

        assert torch.allclose(obs1, obs2, atol=1e-6), "Not reproducible with generator"
        assert torch.equal(reg1, reg2), "Regimes not reproducible with generator"

    def test_regime_switches_use_torch_rng(self):
        """Regime switches should use torch.rand, not numpy"""
        # This test verifies that switching is reproducible with torch seed
        set_seed(42)
        gen = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, p_switch=0.1)
        _, _, reg1 = gen.generate_sequence(T=1000)

        set_seed(42)
        gen = AntipodalRegimeSwitcher(latent_dim=8, obs_dim=4, p_switch=0.1)
        _, _, reg2 = gen.generate_sequence(T=1000)

        assert torch.equal(reg1, reg2), "Regime switches not reproducible"

        # Count switches to verify they're actually happening
        switches1 = (reg1[1:] != reg1[:-1]).sum().item()
        assert switches1 > 0, "No regime switches occurred"


class TestCrossDeviceReproducibility:
    """Test reproducibility across CPU/GPU (if available)"""

    def test_cpu_reproducibility(self):
        """Operations on CPU should be reproducible"""
        set_seed(42)
        x1 = torch.randn(100, device="cpu")

        set_seed(42)
        x2 = torch.randn(100, device="cpu")

        assert torch.equal(x1, x2), "CPU operations not reproducible"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_seed_set(self):
        """CUDA seeds should be set by set_seed"""
        set_seed(42)
        x1 = torch.randn(100, device="cuda")

        set_seed(42)
        x2 = torch.randn(100, device="cuda")

        assert torch.equal(x1, x2), "CUDA operations not reproducible"

    def test_deterministic_flag(self):
        """Deterministic flag should be settable"""
        # Should not raise an error
        set_seed(42, deterministic=True)
        set_seed(42, deterministic=False)


class TestSequenceReproducibility:
    """Test that sequences of operations are reproducible"""

    def test_training_step_reproducibility(self):
        """A simple training step should be reproducible"""
        from src.models import Z2EquivariantRNN

        # First run
        set_seed(42)
        model1 = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        x1 = torch.randn(2, 10, 4)
        y1, _ = model1(x1)
        loss1 = y1.sum()
        loss1.backward()

        # Get gradients
        grad1 = [p.grad.clone() if p.grad is not None else None for p in model1.parameters()]

        # Second run
        set_seed(42)
        model2 = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        x2 = torch.randn(2, 10, 4)
        y2, _ = model2(x2)
        loss2 = y2.sum()
        loss2.backward()

        # Get gradients
        grad2 = [p.grad.clone() if p.grad is not None else None for p in model2.parameters()]

        # Check everything matches
        assert torch.equal(x1, x2), "Inputs not reproducible"
        assert torch.equal(y1, y2), "Outputs not reproducible"
        assert loss1.item() == loss2.item(), "Losses not reproducible"

        for g1, g2 in zip(grad1, grad2):
            if g1 is not None and g2 is not None:
                assert torch.equal(g1, g2), "Gradients not reproducible"
