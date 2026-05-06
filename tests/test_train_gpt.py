import pytest
import os
import sys

# Let real torch and triton load.
import torch

# Just mock out CUDA calls in torch before importing train_gpt.
import torch.cuda
torch.cuda.is_available = lambda: True # pretend it is available to pass the assert!
torch.cuda.set_device = lambda x: None

orig_empty = torch.empty
def mocked_empty(*args, **kwargs):
    if kwargs.get('device') and 'cuda' in str(kwargs.get('device')):
        kwargs.pop('device')
    return orig_empty(*args, **kwargs)
torch.empty = mocked_empty

orig_tensor = torch.tensor
def mocked_tensor(*args, **kwargs):
    if kwargs.get('device') and 'cuda' in str(kwargs.get('device')):
        kwargs.pop('device')
    return orig_tensor(*args, **kwargs)
torch.tensor = mocked_tensor

import torch.distributed as dist
dist.init_process_group = lambda *args, **kwargs: None
dist.barrier = lambda: None
dist.get_world_size = lambda: 8
dist.get_rank = lambda: 0

# Mock only the parts of our custom code that strictly fail
import unittest.mock
class MockTritonKernels:
    FusedLinearReLUSquareFunction = unittest.mock.MagicMock()
    FusedSoftcappedCrossEntropy = unittest.mock.MagicMock()
    XXT = lambda *args, **kwargs: None
    XTX = lambda *args, **kwargs: None
    ba_plus_cAA = lambda *args, **kwargs: None
    transpose_add = lambda *args, **kwargs: None
    transpose_copy = lambda *args, **kwargs: None
sys.modules['triton_kernels'] = MockTritonKernels()
sys.modules['kernels'] = unittest.mock.MagicMock()

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '8'
os.environ['LOCAL_RANK'] = '0'
sys.argv[0] = os.path.abspath('train_gpt.py')

import train_gpt

def test_next_multiple_of_n():
    assert train_gpt.next_multiple_of_n(0, n=5) == 0
    assert train_gpt.next_multiple_of_n(5, n=5) == 5
    assert train_gpt.next_multiple_of_n(10, n=5) == 10

    assert train_gpt.next_multiple_of_n(1, n=5) == 5
    assert train_gpt.next_multiple_of_n(4, n=5) == 5
    assert train_gpt.next_multiple_of_n(6, n=5) == 10
    assert train_gpt.next_multiple_of_n(9, n=5) == 10

    assert train_gpt.next_multiple_of_n(1.5, n=5) == 5
    assert train_gpt.next_multiple_of_n(5.1, n=5) == 10

    assert train_gpt.next_multiple_of_n(-1, n=5) == 0
    assert train_gpt.next_multiple_of_n(-6, n=5) == -5
    assert train_gpt.next_multiple_of_n(-10, n=5) == -10

    with pytest.raises(ZeroDivisionError):
        train_gpt.next_multiple_of_n(5, n=0)

    assert train_gpt.next_multiple_of_n(11, n=3) == 12
    assert train_gpt.next_multiple_of_n(11, n=128) == 128
    assert train_gpt.next_multiple_of_n(129, n=128) == 256

def test_norm():
    import math
    import torch

    # Basic shape test
    x = torch.randn(2, 3, 4)
    y = train_gpt.norm(x)
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"

    # Values test
    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_val = train_gpt.norm(x_val)

    rms0 = math.sqrt((1.0**2 + 2.0**2 + 3.0**2) / 3.0)
    rms1 = math.sqrt((4.0**2 + 5.0**2 + 6.0**2) / 3.0)

    expected_y0 = torch.tensor([1.0 / rms0, 2.0 / rms0, 3.0 / rms0])
    expected_y1 = torch.tensor([4.0 / rms1, 5.0 / rms1, 6.0 / rms1])

    assert torch.allclose(y_val[0], expected_y0, atol=1e-5)
    assert torch.allclose(y_val[1], expected_y1, atol=1e-5)

    # Edge case: zeros
    x_zero = torch.zeros((2, 3))
    y_zero = train_gpt.norm(x_zero)
    # RMS of zeros is 0, F.rms_norm might add eps, but without weight it should be 0s
    assert torch.allclose(y_zero, torch.zeros_like(y_zero))
