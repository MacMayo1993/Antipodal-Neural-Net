"""
Test suite for device handling and GPU compatibility.

Tests that models and parity operators work correctly on both CPU and GPU,
and that all tensors stay on the same device throughout operations.
"""

import pytest
import torch

from src.parity import ParityOperator, ParityProjectors
from src.models import Z2EquivariantRNN, SeamGatedRNN, GRUBaseline


class TestCPUDevice:
    """Test operations on CPU"""

    def test_parity_operator_cpu(self):
        """ParityOperator should work on CPU"""
        parity_op = ParityOperator(dim=8)
        assert parity_op.S.device.type == 'cpu'

        h = torch.randn(8)
        h_transformed = parity_op(h)
        assert h_transformed.device.type == 'cpu'

    def test_parity_projectors_cpu(self):
        """ParityProjectors should work on CPU"""
        parity_op = ParityOperator(dim=8)
        projectors = ParityProjectors(parity_op)

        assert projectors.P_plus.device.type == 'cpu'
        assert projectors.P_minus.device.type == 'cpu'

        h = torch.randn(8)
        h_plus = projectors.project_plus(h)
        h_minus = projectors.project_minus(h)

        assert h_plus.device.type == 'cpu'
        assert h_minus.device.type == 'cpu'

    def test_z2_equivariant_forward_cpu(self):
        """Z2EquivariantRNN forward pass should work on CPU"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        x = torch.randn(2, 10, 4)

        outputs, h = model(x)

        assert outputs.device.type == 'cpu'
        assert h.device.type == 'cpu'

    def test_seam_gated_forward_cpu(self):
        """SeamGatedRNN forward pass should work on CPU"""
        model = SeamGatedRNN(input_dim=4, hidden_dim=8, output_dim=4, gate_type='kstar')
        x = torch.randn(2, 10, 4)

        outputs, h, gates = model(x)

        assert outputs.device.type == 'cpu'
        assert h.device.type == 'cpu'
        assert gates.device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDADevice:
    """Test operations on CUDA"""

    def test_parity_operator_cuda(self):
        """ParityOperator should work on CUDA"""
        parity_op = ParityOperator(dim=8)
        parity_op = parity_op.cuda()

        assert parity_op.S.device.type == 'cuda'

        h = torch.randn(8, device='cuda')
        h_transformed = parity_op(h)
        assert h_transformed.device.type == 'cuda'

    def test_parity_projectors_cuda(self):
        """ParityProjectors should work on CUDA"""
        parity_op = ParityOperator(dim=8)
        projectors = ParityProjectors(parity_op)
        projectors = projectors.cuda()

        assert projectors.P_plus.device.type == 'cuda'
        assert projectors.P_minus.device.type == 'cuda'

        h = torch.randn(8, device='cuda')
        h_plus = projectors.project_plus(h)
        h_minus = projectors.project_minus(h)

        assert h_plus.device.type == 'cuda'
        assert h_minus.device.type == 'cuda'

    def test_z2_equivariant_forward_cuda(self):
        """Z2EquivariantRNN forward pass should work on CUDA"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        model = model.cuda()

        x = torch.randn(2, 10, 4, device='cuda')

        outputs, h = model(x)

        assert outputs.device.type == 'cuda'
        assert h.device.type == 'cuda'

    def test_seam_gated_forward_cuda(self):
        """SeamGatedRNN forward pass should work on CUDA"""
        model = SeamGatedRNN(input_dim=4, hidden_dim=8, output_dim=4, gate_type='kstar')
        model = model.cuda()

        x = torch.randn(2, 10, 4, device='cuda')

        outputs, h, gates = model(x)

        assert outputs.device.type == 'cuda'
        assert h.device.type == 'cuda'
        assert gates.device.type == 'cuda'

    def test_parity_energy_cuda(self):
        """Parity energy computation should work on CUDA"""
        parity_op = ParityOperator(dim=8)
        projectors = ParityProjectors(parity_op)
        projectors = projectors.cuda()

        h = torch.randn(2, 8, device='cuda')
        alpha = projectors.parity_energy(h)

        assert alpha.device.type == 'cuda'
        assert torch.all((alpha >= 0) & (alpha <= 1)), "Parity energy out of bounds"


class TestBuffersMovedWithModel:
    """Test that buffers move correctly with model"""

    def test_parity_buffers_move_with_model(self):
        """Buffers should move when model is moved"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)

        # Check initial device
        assert model.parity_op.S.device.type == 'cpu'

        # Move to CPU explicitly (should be no-op but safe)
        model = model.cpu()
        assert model.parity_op.S.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_parity_buffers_move_to_cuda(self):
        """Buffers should move to CUDA with model"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)

        # Move to CUDA
        model = model.cuda()

        # Check buffers moved
        assert model.parity_op.S.device.type == 'cuda'
        assert model.projectors.P_plus.device.type == 'cuda'
        assert model.projectors.P_minus.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_seam_gated_buffers_move_to_cuda(self):
        """SeamGatedRNN buffers should move to CUDA"""
        model = SeamGatedRNN(input_dim=4, hidden_dim=8, output_dim=4, gate_type='kstar')

        # Move to CUDA
        model = model.cuda()

        # Check S buffer (registered in forward)
        assert model.S.device.type == 'cuda'


class TestMixedDeviceErrors:
    """Test that mixed device operations fail gracefully"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_model_cuda_input_fails(self):
        """CPU model with CUDA input should raise error"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        x = torch.randn(2, 10, 4, device='cuda')

        with pytest.raises(RuntimeError):
            model(x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_model_cpu_input_fails(self):
        """CUDA model with CPU input should raise error"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        model = model.cuda()

        x = torch.randn(2, 10, 4, device='cpu')

        with pytest.raises(RuntimeError):
            model(x)


class TestDeviceConsistency:
    """Test that all tensors stay on same device"""

    def test_forward_pass_device_consistency(self):
        """All outputs should be on same device as inputs"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        x = torch.randn(2, 10, 4)

        outputs, h = model(x)

        assert outputs.device == x.device
        assert h.device == x.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass_device_consistency_cuda(self):
        """All outputs should be on same device as inputs (CUDA)"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        model = model.cuda()

        x = torch.randn(2, 10, 4, device='cuda')

        outputs, h = model(x)

        assert outputs.device == x.device
        assert h.device == x.device

    def test_gate_device_consistency(self):
        """Gates should be on same device as inputs"""
        model = SeamGatedRNN(input_dim=4, hidden_dim=8, output_dim=4, gate_type='fixed')
        x = torch.randn(2, 10, 4)

        outputs, h, gates = model(x)

        assert gates.device == x.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gate_device_consistency_cuda(self):
        """Gates should be on same device as inputs (CUDA)"""
        model = SeamGatedRNN(input_dim=4, hidden_dim=8, output_dim=4, gate_type='fixed')
        model = model.cuda()

        x = torch.randn(2, 10, 4, device='cuda')

        outputs, h, gates = model(x)

        assert gates.device == x.device


class TestNoDeviceHardcoding:
    """Test that no device is hardcoded"""

    def test_zero_tensors_respect_input_device(self):
        """Zero tensors should respect input device"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)

        # Test with None hidden state on CPU
        x = torch.randn(2, 10, 4)
        outputs, h = model(x, h=None)

        assert h.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_zero_tensors_respect_input_device_cuda(self):
        """Zero tensors should respect input device (CUDA)"""
        model = Z2EquivariantRNN(input_dim=4, hidden_dim=8, output_dim=4)
        model = model.cuda()

        # Test with None hidden state on CUDA
        x = torch.randn(2, 10, 4, device='cuda')
        outputs, h = model(x, h=None)

        assert h.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_parity_energy_respects_device(self):
        """Parity energy zeros should respect input device"""
        parity_op = ParityOperator(dim=8)
        projectors = ParityProjectors(parity_op)
        projectors = projectors.cuda()

        # Test with zero norm
        h = torch.zeros(2, 8, device='cuda')
        alpha = projectors.parity_energy(h)

        assert alpha.device.type == 'cuda'
