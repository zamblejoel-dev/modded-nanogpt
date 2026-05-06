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

    # Shape tests
    for shape in [(10,), (2, 5), (3, 4, 5), (2, 3, 4, 5)]:
        x = torch.randn(shape)
        y = norm(x)
        assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"

    # Dtype tests
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        x = torch.randn((2, 3, 4), dtype=dtype)
        y = norm(x)
        assert y.dtype == dtype, f"Expected dtype {dtype}, got {y.dtype}"
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

    # Independent dimension tests
    # Modifying the first row should not affect the second row
    x_indep = torch.randn(2, 3, dtype=torch.float32)
    y_indep = norm(x_indep)
    x_indep_modified = x_indep.clone()
    x_indep_modified[0] = x_indep_modified[0] * 2.0
    y_indep_modified = norm(x_indep_modified)
    assert torch.allclose(y_indep[1], y_indep_modified[1], atol=1e-5), "Norm is not independent across rows"

    # Manual implementation test
    x_manual = torch.randn(10, 20, 30, dtype=torch.float32)
    y_manual = norm(x_manual)
    # F.rms_norm default eps is 1e-5
    y_expected = x_manual * torch.rsqrt(x_manual.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(y_manual, y_expected, atol=1e-4)


def test_norm_train_gpt_medium():
    norm = get_norm_func("train_gpt_medium.py")
    if norm is None:
        pytest.skip("norm function not found in train_gpt_medium.py")

    # Shape tests
    for shape in [(10,), (2, 5), (3, 4, 5), (2, 3, 4, 5)]:
        x = torch.randn(shape)
        y = norm(x)
        assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"

    # Dtype tests
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        x = torch.randn((2, 3, 4), dtype=dtype)
        y = norm(x)
        assert y.dtype == dtype, f"Expected dtype {dtype}, got {y.dtype}"
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

    # Independent dimension tests
    # Modifying the first row should not affect the second row
    x_indep = torch.randn(2, 3, dtype=torch.float32)
    y_indep = norm(x_indep)
    x_indep_modified = x_indep.clone()
    x_indep_modified[0] = x_indep_modified[0] * 2.0
    y_indep_modified = norm(x_indep_modified)
    assert torch.allclose(y_indep[1], y_indep_modified[1], atol=1e-5), "Norm is not independent across rows"

    # Manual implementation test
    x_manual = torch.randn(10, 20, 30, dtype=torch.float32)
    y_manual = norm(x_manual)
    # F.rms_norm default eps is 1e-5
    y_expected = x_manual * torch.rsqrt(x_manual.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(y_manual, y_expected, atol=1e-4)
