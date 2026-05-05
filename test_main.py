import torch
import sys
import os

os.environ['LOCAL_RANK'] = '0'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

sys.argv[0] = 'train_gpt.py'
import train_gpt

def test_bigram_hash():
    x = torch.randint(0, 50257, (16*2048,), dtype=torch.int32)
    # verify it runs without crashing
    out = train_gpt.get_bigram_hash(x)
    assert out.shape == x.shape
    assert out.dtype == torch.int32
