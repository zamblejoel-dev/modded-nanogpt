import pytest
import torch
import torch.nn.functional as F
import math

# We create a dummy environment to evaluate train_gpt.py
# without triggering distributed or triton imports where possible
# Actually, train_gpt.py runs distributed initialization unconditionally at module scope.
# The previous approach using `exec` to extract `norm` was good, but let's improve it.

def get_norm_func(filename):
    namespace = {'Tensor': torch.Tensor, 'F': F}
    with open(filename, "r") as f:
        content = f.read()

    # We can parse the file and find the function definition
    import ast
    tree = ast.parse(content)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'norm':
            # reconstruct the function string and exec
            func_str = ast.unparse(node)
            exec(func_str, namespace)
            return namespace['norm']
    return None

def test_norm_train_gpt():
    norm = get_norm_func("train_gpt.py")
    assert norm is not None

    x = torch.randn(2, 3, 4)
    y = norm(x)
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_val = norm(x_val)

    rms0 = math.sqrt((1.0**2 + 2.0**2 + 3.0**2) / 3.0)
    rms1 = math.sqrt((4.0**2 + 5.0**2 + 6.0**2) / 3.0)

    expected_y0 = torch.tensor([1.0 / rms0, 2.0 / rms0, 3.0 / rms0])
    expected_y1 = torch.tensor([4.0 / rms1, 5.0 / rms1, 6.0 / rms1])

    assert torch.allclose(y_val[0], expected_y0, atol=1e-5)
    assert torch.allclose(y_val[1], expected_y1, atol=1e-5)

    x_zero = torch.zeros((2, 3))
    y_zero = norm(x_zero)
    assert torch.allclose(y_zero, torch.zeros_like(y_zero))

def test_norm_train_gpt_medium():
    norm = get_norm_func("train_gpt_medium.py")
    if norm is None:
        pytest.skip("norm function not found in train_gpt_medium.py")

    x = torch.randn(2, 3, 4)
    y = norm(x)
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_val = norm(x_val)

    rms0 = math.sqrt((1.0**2 + 2.0**2 + 3.0**2) / 3.0)
    rms1 = math.sqrt((4.0**2 + 5.0**2 + 6.0**2) / 3.0)

    expected_y0 = torch.tensor([1.0 / rms0, 2.0 / rms0, 3.0 / rms0])
    expected_y1 = torch.tensor([4.0 / rms1, 5.0 / rms1, 6.0 / rms1])

    assert torch.allclose(y_val[0], expected_y0, atol=1e-5)
    assert torch.allclose(y_val[1], expected_y1, atol=1e-5)

    x_zero = torch.zeros((2, 3))
    y_zero = norm(x_zero)
    assert torch.allclose(y_zero, torch.zeros_like(y_zero))
