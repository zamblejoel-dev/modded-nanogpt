import os
import sys
import pytest
import torch
import torch.nn.functional as F

# It is proving too difficult to safely mock all of the CUDA initialization and training
# loop code that is present at the module level in train_gpt.py.
# However, the `norm` function is completely pure and has zero side effects or external dependencies.
# In a real environment, we would refactor train_gpt.py to not execute global training code on import
# (by putting it in a `if __name__ == '__main__':` block).
# As a workaround, we will temporarily modify our mock to simulate reading the file
# and executing just the norm function, or we can use `ast` or `exec` to extract it.
# Actually, an easier way is to just define it directly since it's just a one-liner wrapper
# around F.rms_norm and the problem statement asks us to test it. Wait, the goal is to test the actual code.
# Let's extract the function text and exec it.

train_gpt_path = os.path.join(os.path.dirname(__file__), '..', 'train_gpt.py')
with open(train_gpt_path, 'r') as f:
    lines = f.readlines()

# Extract the norm function definition
in_norm = False
norm_code = []
for line in lines:
    if line.startswith('def norm(x: Tensor):'):
        in_norm = True
        norm_code.append('from torch import Tensor\n')
        norm_code.append(line)
    elif in_norm:
        if line.strip() == '' or line.startswith(' '):
            norm_code.append(line)
        else:
            break

# Execute the extracted function in the local namespace
exec(''.join(norm_code))

def test_norm_correctness():
    """Test that norm function produces correct RMSNorm output values"""
    x = torch.randn(2, 3, 4)
    out = norm(x)
    expected = F.rms_norm(x, (x.size(-1),))
    assert torch.allclose(out, expected)

def test_norm_shape():
    """Test that norm function preserves tensor shape"""
    x = torch.randn(2, 3, 4)
    out = norm(x)
    assert out.shape == x.shape

def test_norm_1d():
    """Test that norm function works with 1D tensors"""
    x = torch.randn(5)
    out = norm(x)
    expected = F.rms_norm(x, (x.size(-1),))
    assert torch.allclose(out, expected)

def test_norm_type():
    """Test that norm function preserves tensor dtype"""
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    out = norm(x)
    assert out.dtype == torch.float16

def test_norm_large_values():
    """Test that norm handles large values without immediate overflow"""
    x = torch.randn(2, 4) * 1000
    out = norm(x)
    expected = F.rms_norm(x, (x.size(-1),))
    assert torch.allclose(out, expected)
