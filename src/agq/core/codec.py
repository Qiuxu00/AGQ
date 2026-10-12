"""GHC1 grouped canonical Huffman, compatible with the CODEBOOK003 C++ format.

The Python path is an inspectable CPU reference; use the optional C++ tool for
large weights. W1 stays raw; W2 groups four codes and W3/W4 group two codes.
Selection counts the full header, table and chunk offsets. Metadata is external
and unchanged. The one-byte selector belongs to both raw and coded candidates.
"""
from collections import Counter
import hashlib
import heapq
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


def _book(lengths):
    ordered = sorted((n, s) for s, n in enumerate(lengths) if n)
    if not ordered or any(n > 24 for n, _ in ordered):
        raise ValueError("invalid code length")
    result, code, previous = {}, 0, 0
    for width, symbol in ordered:
        code <<= width - previous
        if code >= 1 << width:
            raise ValueError("invalid prefix lengths")
        reversed_code = int(f"{code:0{width}b}"[::-1], 2)
        result[symbol] = (reversed_code, width)
        code, previous = code + 1, width
    return result


def _symbols(raw, width):
    accumulator = available = 0
    for value in raw:
        accumulator |= value << available
        available += 8
        while available >= width:
            yield accumulator & ((1 << width) - 1)
            accumulator >>= width
            available -= width
    if available:
        raise ValueError("unaligned raw input")


def encode_ghc1(raw, width, block_bytes=16384):
    """Encode the original GHC1 binary format; width is grouped symbol width."""
    raw = bytes(raw)
    if width not in (1, 2, 3, 4, 6, 8) or not raw or len(raw) * 8 % width:
        raise ValueError("nonempty aligned data and a supported symbol width required")
    if type(block_bytes) is not int or not 0 < block_bytes <= 1048576:
        raise ValueError("invalid block size")
    if width in (3, 6):
        block_bytes -= block_bytes % 3
    if not block_bytes:
        raise ValueError("empty aligned block")
    counts = Counter(_symbols(raw, width))
    nodes, heap = [], []
    for symbol, count in sorted(counts.items()):
        heapq.heappush(heap, (count, len(nodes)))
        nodes.append((None, None, symbol))
    while len(heap) > 1:
        count_a, a = heapq.heappop(heap)
        count_b, b = heapq.heappop(heap)
        heapq.heappush(heap, (count_a + count_b, len(nodes)))
        nodes.append((a, b, None))
    lengths = [0] * max(16, 1 << width)
    pending = [(heap[0][1], 0)]
    while pending:
        index, depth = pending.pop()
        a, b, symbol = nodes[index]
        if symbol is None:
            pending.extend(((a, depth + 1), (b, depth + 1)))
        else:
            lengths[symbol] = max(1, depth)
    codes = _book(lengths)
    stream, offsets = bytearray(), [0]
    for start in range(0, len(raw), block_bytes):
        accumulator = available = 0
        for symbol in _symbols(raw[start:start + block_bytes], width):
            value, nbits = codes[symbol]
            accumulator |= value << available
            available += nbits
            while available >= 8:
                stream.append(accumulator & 255)
                accumulator >>= 8
                available -= 8
        if available:
            stream.append(accumulator)
        offsets.append(len(stream))
    return (struct.pack("<4sIQQB", b"GHC1", block_bytes, len(raw), len(offsets) - 1, width)
            + bytes(lengths) + struct.pack("<" + "Q" * len(offsets), *offsets) + stream)


def parse_ghc1(blob, *, max_output_bytes=1 << 32):
    """Validate framing before allocation or CUDA buffer construction."""
    blob = bytes(blob)
    if len(blob) < 25:
        raise ValueError("truncated GHC1 header")
    magic, block, size, chunks, width = struct.unpack_from("<4sIQQB", blob)
    if magic != b"GHC1" or not 0 < block <= 1048576 or not 0 < size <= max_output_bytes:
        raise ValueError("invalid GHC1 dimensions")
    if width not in (1, 2, 3, 4, 6, 8) or size * 8 % width or block * 8 % width:
        raise ValueError("invalid symbol alignment")
    if chunks != (size + block - 1) // block:
        raise ValueError("invalid chunk count")
    pos = 25 + max(16, 1 << width)
    end = pos + 8 * (chunks + 1)
    if end > len(blob):
        raise ValueError("truncated GHC1 metadata")
    lengths = list(blob[25:pos])
    _book(lengths)
    offsets = list(struct.unpack_from("<" + "Q" * (chunks + 1), blob, pos))
    data = blob[end:]
    if offsets[0] != 0 or offsets[-1] != len(data) or any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("invalid stream offsets")
    return {"block": block, "size": size, "width": width, "lengths": lengths,
            "offsets": offsets, "data": data}


def decode_ghc1(blob, *, max_output_bytes=1 << 32):
    p = parse_ghc1(blob, max_output_bytes=max_output_bytes)
    codes = _book(p["lengths"])
    lookup = {(value, length): symbol for symbol, (value, length) in codes.items()}
    maxwidth = max(p["lengths"])
    output = bytearray()
    for chunk, (start, end) in enumerate(zip(p["offsets"], p["offsets"][1:])):
        source = p["data"][start:end]
        pos = accumulator = available = 0
        count = min(p["block"], p["size"] - chunk * p["block"]) * 8 // p["width"]
        for _ in range(count):
            value = 0
            for length in range(1, maxwidth + 1):
                if pos >= len(source) * 8:
                    raise ValueError("truncated compressed symbol")
                value |= ((source[pos // 8] >> (pos % 8)) & 1) << (length - 1)
                pos += 1
                symbol = lookup.get((value, length))
                if symbol is not None:
                    break
            else:
                raise ValueError("invalid compressed symbol")
            if symbol >= 1 << p["width"]:
                raise ValueError("symbol exceeds source width")
            accumulator |= symbol << available
            available += p["width"]
            while available >= 8:
                output.append(accumulator & 255)
                accumulator >>= 8
                available -= 8
        padding = len(source) * 8 - pos
        if available or padding >= 8 or (padding and source[-1] >> (8 - padding)):
            raise ValueError("trailing data or nonzero padding")
    return bytes(output)


def build_native(output_dir=None, compiler="g++"):
    """Compile the included original C++ codec to a user cache, never source dirs."""
    source = Path(__file__).with_name("native") / "group_huffman.cpp"
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(f"C++ compiler unavailable: {compiler}; use backend='python'")
    folder = (Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "agq-codec").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    import os
    filename = "group_huffman_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = folder / (filename + (".exe" if os.name == "nt" else ""))
    if not target.exists():
        # Relative compiler output plus source on stdin also works with MinGW
        # toolchains whose narrow-character linker cannot open Unicode paths.
        subprocess.run([compiler_path, "-std=c++17", "-O3", "-x", "c++", "-", "-o", target.name],
                       input=source.read_bytes(), cwd=folder, check=True)
    return target


def encode(raw, bits, *, backend="python", executable=None):
    """Return selector+payload and a storage/roundtrip report (CODEBOOK003)."""
    raw = bytes(raw)
    if bits not in (1, 2, 3, 4) or backend not in ("python", "native"):
        raise ValueError("W1--W4 and python/native backend required")
    candidate, rejected = None, None
    width = bits * (4 if bits == 2 else 2)
    if bits > 1 and raw:
        if len(raw) * 8 % width:
            rejected = "group alignment requires raw fallback"
        else:
            try:
                if backend == "native":
                    tool = executable or build_native()
                    result = subprocess.run([str(tool), "encode", "16384", str(width)], input=raw, capture_output=True)
                    if result.returncode:
                        raise ValueError(result.stderr.decode(errors="replace").strip())
                    candidate = result.stdout
                else:
                    candidate = encode_ghc1(raw, width)
            except ValueError as exc:
                if str(exc) != "invalid code length":
                    raise
                rejected = str(exc)
    use_coded = candidate is not None and len(candidate) < len(raw)
    blob = b"\x03" + candidate if use_coded else b"\x00" + raw
    if decode(blob) != raw:
        raise RuntimeError("lossless codec roundtrip failed")
    return blob, {"mode": "group_huffman" if use_coded else "raw", "raw_bytes": len(raw),
                  "stored_bytes": len(blob), "selector_bytes": 1,
                  "candidate_bytes": {"raw": len(raw), "group_huffman": None if candidate is None else len(candidate)},
                  "raw_sha256": hashlib.sha256(raw).hexdigest(),
                  "stored_sha256": hashlib.sha256(blob).hexdigest(), "roundtrip_exact": True,
                  "rejected_candidate": rejected}


def decode(blob, *, max_output_bytes=1 << 32):
    blob = bytes(blob)
    if not blob:
        raise ValueError("missing codec selector")
    if blob[0] == 0:
        if len(blob) - 1 > max_output_bytes:
            raise ValueError("raw payload exceeds output limit")
        return blob[1:]
    if blob[0] == 3:
        return decode_ghc1(blob[1:], max_output_bytes=max_output_bytes)
    raise ValueError("only raw and CODEBOOK003 grouped Huffman are supported")
