"""Original CODEBOOK003 CUDA decoder with validated GHC1 input framing."""
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from agq.core.codec import parse_ghc1, decode_ghc1


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    return load(name="agq_ghc1_cuda", sources=[str(Path(__file__).with_name("entropy_cuda.cu"))],
                extra_cuda_cflags=["-O3", "-Xcompiler", "-fopenmp"],
                extra_ldflags=["-lz", "-fopenmp"], verbose=False)


def lookup_tables(lengths):
    missing, children, code, previous = -2147483647, [[-2147483647, -2147483647]], 0, 0
    for length, symbol in sorted((n, s) for s, n in enumerate(lengths) if n):
        code <<= length - previous
        node = 0
        for shift in range(length - 1, -1, -1):
            branch = (code >> shift) & 1
            if shift == 0:
                if children[node][branch] != missing:
                    raise ValueError("prefix overlap")
                children[node][branch] = -symbol - 1
            else:
                if children[node][branch] == missing:
                    children[node][branch] = len(children)
                    children.append([missing, missing])
                node = children[node][branch]
                if node < 0:
                    raise ValueError("prefix overlap")
        code += 1
        previous = length
    root = []
    for prefix in range(1024):
        node = entry = 0
        for depth in range(10):
            node = children[node][(prefix >> depth) & 1]
            if node == missing:
                break
            if node < 0:
                entry = ((depth + 1) << 24) | (-node - 1)
                break
        else:
            entry = -node - 1
        root.append(entry)
    return root, [v for pair in children for v in pair]


class GPUHuffman(nn.Module):
    def __init__(self, blob, device):
        super().__init__()
        if torch.device(device).type != "cuda":
            raise ValueError("CUDA device required")
        p = parse_ghc1(blob)
        # Validate the entire stream before invoking the original GPU extension.
        reference = decode_ghc1(blob)
        root, children = lookup_tables(p["lengths"])
        for name, values, dtype in [("data", list(p["data"]), torch.uint8), ("offsets", p["offsets"], torch.int64),
                                     ("root", root, torch.int32), ("children", children, torch.int32)]:
            self.register_buffer(name, torch.tensor(values, dtype=dtype, device=device))
        self.size, self.block, self.width = p["size"], p["block"], p["width"]
        checkpoints = extension().build_checkpoints(self.data.cpu(), self.offsets.cpu(), self.root.cpu(), self.children.cpu(), self.size, self.block, self.width)
        self.register_buffer("checkpoints", checkpoints.to(device))
        if self.decode().cpu().numpy().tobytes() != reference:
            raise RuntimeError("CUDA code-stream verification failed")

    def decode(self):
        return extension().decode(self.data, self.offsets, self.root, self.children,
                                  self.checkpoints, self.size, self.block, self.width)
