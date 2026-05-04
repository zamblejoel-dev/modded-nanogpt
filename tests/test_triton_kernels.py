import sys
from unittest.mock import MagicMock
import pytest

# Attempt to import real libraries
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = MagicMock()
    torch.float32 = "float32"
    torch.bfloat16 = "bfloat16"
    torch.cuda.is_available.return_value = False
    sys.modules["torch"] = torch

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = MagicMock()
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = MagicMock()
    sys.modules["triton.tools.tensor_descriptor"] = MagicMock()

# Import the function to test
from triton_kernels import ba_plus_cAA

def torch_ba_plus_cAA(A, alpha, beta):
    """Reference implementation using standard PyTorch."""
    # C = alpha * A @ A.T + beta * A
    # A.T is A.transpose(-1, -2)
    return alpha * (A @ A.transpose(-1, -2).to(torch.float32)).to(A.dtype) + beta * A

@pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA or PyTorch not available")
@pytest.mark.parametrize("shape", [(128, 128), (2, 256, 256)])
@pytest.mark.parametrize("alpha", [1.0, 0.5])
@pytest.mark.parametrize("beta", [0.0, 1.0, -0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ba_plus_cAA_correctness(shape, alpha, beta, dtype):
    """Verify correctness by comparing against PyTorch reference implementation."""
    device = "cuda"
    A = torch.randn(shape, device=device, dtype=dtype)
    out_triton = torch.empty(shape, device=device, dtype=dtype)

    ba_plus_cAA(A, alpha, beta, out_triton)
    out_torch = torch_ba_plus_cAA(A, alpha, beta)

    # Check result
    if dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5

    torch.testing.assert_close(out_triton, out_torch, atol=atol, rtol=rtol)

def test_ba_plus_cAA_assertions():
    """Verify that the wrapper correctly asserts on invalid inputs."""
    if HAS_TORCH:
        # Use real CPU tensors if possible
        A = torch.randn(128, 64)
        out = torch.empty(128, 128)
    else:
        # Fallback to mocks
        A = MagicMock()
        A.ndim = 2
        A.shape = (128, 64)
        A.size.side_effect = lambda i: A.shape[i]
        out = MagicMock()
        out.size.return_value = 128
        out.ndim = 2

    # Test non-square matrix
    with pytest.raises(AssertionError, match="Input matrix must be square"):
        ba_plus_cAA(A, 1.0, 1.0, out)

    # Test incorrect output shape
    if HAS_TORCH:
        A = torch.randn(128, 128)
        out = torch.empty(128, 64)
    else:
        A.shape = (128, 128)
        out.size.return_value = 64

    with pytest.raises(AssertionError):
        ba_plus_cAA(A, 1.0, 1.0, out)

def test_ba_plus_cAA_ndim_assertion():
    """Verify that the wrapper correctly asserts on invalid dimensionality."""
    if HAS_TORCH:
        A = torch.randn(128)
        out = torch.empty(128, 128)
    else:
        A = MagicMock()
        A.ndim = 1
        out = MagicMock()

    with pytest.raises(AssertionError):
        ba_plus_cAA(A, 1.0, 1.0, out)

def test_ba_plus_cAA_kernel_call():
    """Verify that the wrapper actually attempts to launch the Triton kernel."""
    import triton_kernels

    # Mock the kernel
    original_kernel = triton_kernels.ba_plus_cAA_kernel
    triton_kernels.ba_plus_cAA_kernel = MagicMock()

    try:
        if HAS_TORCH:
            A = torch.randn(128, 128)
            out = torch.empty(128, 128)
        else:
            A = MagicMock()
            A.ndim = 2
            A.shape = (128, 128)
            A.size.return_value = 128
            A.stride.return_value = 1
            out = MagicMock()
            out.ndim = 2
            out.size.return_value = 128
            out.stride.return_value = 1

        ba_plus_cAA(A, 1.0, 1.0, out)

        # Check if kernel was called
        # Triton kernels are called as kernel[grid](args)
        assert triton_kernels.ba_plus_cAA_kernel.called or triton_kernels.ba_plus_cAA_kernel.__getitem__.called
    finally:
        # Restore original kernel
        triton_kernels.ba_plus_cAA_kernel = original_kernel
