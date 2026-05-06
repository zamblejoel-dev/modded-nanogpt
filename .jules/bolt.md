## 2024-03-08 - CPU Bottleneck in Data Loader via Eager execution
**Learning:** PyTorch eager tensor operations in the data loader iteration (e.g. computing bigram hashing for each batch position) can create a substantial CPU overhead when executed thousands of times. `torch.compile` is highly effective but must be wrapped correctly because compiling code with `pin_memory=True` directly doesn't usually map well to fullgraph compilation.
**Action:** Use `torch.compile` for mathematical preprocessing operations in data loaders but extract host-specific memory layout modifiers like `pin_memory()` to the eager wrapping function.

## 2026-05-06 - Added torch.compile to bigram hash
**Learning:** The wrapper function separated `pin_memory` from the compiled function properly, but the `@torch.compile(fullgraph=True)` decorator was missing from the inner function, leading to slow eager execution. Adding it sped up execution significantly (from 0.4s to 0.08s for 100 iterations of batch processing in my benchmark).
**Action:** Always check that the inner function targeted for compilation actually has the decorator applied when using eager wrapper patterns.
