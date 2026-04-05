#!/usr/bin/env python3
"""
Convert WeSpeaker embedding checkpoint to GGUF.

The output naming and metadata are aligned with the current C++ loader in
`src/pyannote/embedding.cpp`.

Usage:
    python examples/python/convert_embedding_to_ggml.py \
        /path/to/embedding/pytorch_model.bin \
        /path/to/pyannote-embedding.gguf
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8


def write_gguf_string(f, value: str):
    encoded = value.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def write_metadata_kv(f, key: str, value_type: int, value):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", value_type))

    if value_type == GGUF_TYPE_UINT32:
        f.write(struct.pack("<I", value))
    elif value_type == GGUF_TYPE_STRING:
        write_gguf_string(f, value)
    else:
        raise ValueError(f"unsupported metadata type: {value_type}")


def align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return offset + (alignment - (offset % alignment)) % alignment


def load_state_dict(model_path: str):
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    return checkpoint


def should_use_f16(name: str, data: np.ndarray) -> bool:
    if "bias" in name:
        return False
    if "running_mean" in name or "running_var" in name:
        return False
    if "bn" in name or "shortcut.1" in name or "shortcut.3" in name:
        if len(data.shape) == 1:
            return False
    if len(data.shape) < 2:
        return False
    return True


def write_gguf(output_path: str, tensors: dict):
    metadata = {
        "general.architecture": (GGUF_TYPE_STRING, "wespeaker_resnet34"),
        "general.name": (GGUF_TYPE_STRING, "pyannote-embedding-community-1"),
        "general.alignment": (GGUF_TYPE_UINT32, GGUF_DEFAULT_ALIGNMENT),
        "wespeaker.sample_rate": (GGUF_TYPE_UINT32, 16000),
        "wespeaker.num_mel_bins": (GGUF_TYPE_UINT32, 80),
        "wespeaker.frame_length": (GGUF_TYPE_UINT32, 25),
        "wespeaker.frame_shift": (GGUF_TYPE_UINT32, 10),
        "wespeaker.embed_dim": (GGUF_TYPE_UINT32, 256),
        "wespeaker.feat_dim": (GGUF_TYPE_UINT32, 80),
    }

    tensor_infos = []
    current_offset = 0
    for name, data in tensors.items():
        ftype = GGML_TYPE_F16 if should_use_f16(name, data) else GGML_TYPE_F32
        converted = np.ascontiguousarray(
            data.astype(np.float16 if ftype == GGML_TYPE_F16 else np.float32)
        )
        tensor_infos.append(
            {
                "name": name,
                "data": converted,
                "ftype": ftype,
                "offset": current_offset,
            }
        )
        current_offset += align_offset(converted.nbytes)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensor_infos)))
        f.write(struct.pack("<Q", len(metadata)))

        for key, (value_type, value) in metadata.items():
            write_metadata_kv(f, key, value_type, value)

        for info in tensor_infos:
            write_gguf_string(f, info["name"])
            data = info["data"]
            n_dims = len(data.shape)
            f.write(struct.pack("<I", n_dims))
            for i in range(n_dims):
                f.write(struct.pack("<Q", data.shape[n_dims - 1 - i]))
            f.write(struct.pack("<I", info["ftype"]))
            f.write(struct.pack("<Q", info["offset"]))

        current_pos = f.tell()
        aligned_pos = align_offset(current_pos)
        if aligned_pos > current_pos:
            f.write(b"\x00" * (aligned_pos - current_pos))

        tensor_data_start = f.tell()
        for info in tensor_infos:
            data = info["data"]
            target_pos = tensor_data_start + info["offset"]
            current_pos = f.tell()
            if target_pos > current_pos:
                f.write(b"\x00" * (target_pos - current_pos))
            data.tofile(f)
            padding = align_offset(data.nbytes) - data.nbytes
            if padding > 0:
                f.write(b"\x00" * padding)

    return tensor_infos


def main():
    parser = argparse.ArgumentParser(description="Convert WeSpeaker embedding checkpoint to GGUF")
    parser.add_argument("model_path", help="Path to PyTorch checkpoint, usually embedding/pytorch_model.bin")
    parser.add_argument("output_path", help="Path to output GGUF file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce logging")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedding checkpoint: {model_path}")
    state_dict = load_state_dict(str(model_path))
    print(f"Found {len(state_dict)} raw tensors")

    converted = {}
    skipped = []
    verbose = not args.quiet
    for name, tensor in state_dict.items():
        if "num_batches_tracked" in name:
            skipped.append(name)
            continue

        data = tensor.detach().cpu().numpy()
        converted[name] = data
        if verbose:
            dtype_str = "F16" if should_use_f16(name, data) else "F32"
            print(f"  {name:55s} {str(tuple(data.shape)):20s} -> {dtype_str}")

    infos = write_gguf(args.output_path, converted)
    f16_count = sum(1 for info in infos if info["ftype"] == GGML_TYPE_F16)
    f32_count = len(infos) - f16_count

    print(f"Wrote {args.output_path}")
    print(f"  tensors: {len(infos)}")
    print(f"  skipped num_batches_tracked: {len(skipped)}")
    print(f"  f16: {f16_count}")
    print(f"  f32: {f32_count}")


if __name__ == "__main__":
    main()
