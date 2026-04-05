#!/usr/bin/env python3
"""
Convert PyAnnote segmentation model from PyTorch checkpoint to GGUF.

The output naming and metadata are aligned with the current C++ loader in
`src/pyannote/segmentation.cpp`.

Usage:
    python examples/python/convert_pyannote_to_ggml.py \
        /path/to/segmentation/pytorch_model.bin \
        /path/to/pyannote-segmentation.gguf
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
GGUF_TYPE_INT32 = 5
GGUF_TYPE_BOOL = 7
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
    elif value_type == GGUF_TYPE_INT32:
        f.write(struct.pack("<i", value))
    elif value_type == GGUF_TYPE_BOOL:
        f.write(struct.pack("B", 1 if value else 0))
    elif value_type == GGUF_TYPE_STRING:
        write_gguf_string(f, value)
    else:
        raise ValueError(f"unsupported metadata type: {value_type}")


def align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return offset + (alignment - (offset % alignment)) % alignment


def load_state_dict(model_path: str):
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    return checkpoint


def extract_sincnet_filters(state_dict, stride: int, sample_rate: int) -> np.ndarray:
    from asteroid_filterbanks import ParamSincFB

    fb_prefix = "sincnet.conv1d.0.filterbank."
    fb_state = {}
    for key, value in state_dict.items():
        if key.startswith(fb_prefix):
            fb_state[key[len(fb_prefix):]] = value

    if not fb_state:
        raise ValueError("could not find SincNet filterbank parameters in state_dict")

    filterbank = ParamSincFB(
        n_filters=80,
        kernel_size=251,
        stride=stride,
        sample_rate=sample_rate,
        min_low_hz=50,
        min_band_hz=50,
    )
    filterbank.load_state_dict(fb_state, strict=True)

    filters = filterbank.filters() if callable(filterbank.filters) else filterbank.filters
    return filters.detach().cpu().numpy()


def map_segmentation_tensors(state_dict, verbose: bool):
    converted = {}

    converted["sincnet.0.conv.weight"] = extract_sincnet_filters(
        state_dict, stride=10, sample_rate=16000
    )

    for name, tensor in state_dict.items():
        if "filterbank.low_hz_" in name or "filterbank.band_hz_" in name:
            continue

        data = tensor.detach().cpu().numpy()
        gguf_name = None

        if name == "sincnet.wav_norm1d.weight":
            gguf_name = "sincnet.wav_norm.weight"
        elif name == "sincnet.wav_norm1d.bias":
            gguf_name = "sincnet.wav_norm.bias"
        elif name == "sincnet.conv1d.1.weight":
            gguf_name = "sincnet.1.conv.weight"
        elif name == "sincnet.conv1d.1.bias":
            gguf_name = "sincnet.1.conv.bias"
        elif name == "sincnet.conv1d.2.weight":
            gguf_name = "sincnet.2.conv.weight"
        elif name == "sincnet.conv1d.2.bias":
            gguf_name = "sincnet.2.conv.bias"
        elif name == "sincnet.norm1d.0.weight":
            gguf_name = "sincnet.0.norm.weight"
        elif name == "sincnet.norm1d.0.bias":
            gguf_name = "sincnet.0.norm.bias"
        elif name == "sincnet.norm1d.1.weight":
            gguf_name = "sincnet.1.norm.weight"
        elif name == "sincnet.norm1d.1.bias":
            gguf_name = "sincnet.1.norm.bias"
        elif name == "sincnet.norm1d.2.weight":
            gguf_name = "sincnet.2.norm.weight"
        elif name == "sincnet.norm1d.2.bias":
            gguf_name = "sincnet.2.norm.bias"
        elif name.startswith("lstm."):
            gguf_name = name
        elif name in {
            "linear.0.weight",
            "linear.0.bias",
            "linear.1.weight",
            "linear.1.bias",
            "classifier.weight",
            "classifier.bias",
        }:
            gguf_name = name

        if gguf_name is None:
            if verbose:
                print(f"Skipping unrecognized tensor: {name}")
            continue

        converted[gguf_name] = data
        if verbose:
            print(f"  {name:40s} -> {gguf_name:28s} {tuple(data.shape)}")

    return converted


def choose_ftype(name: str, data: np.ndarray) -> int:
    if "bias" in name or len(data.shape) == 1:
        return GGML_TYPE_F32
    return GGML_TYPE_F16


def write_gguf(output_path: str, tensors: dict):
    metadata = {
        "general.architecture": (GGUF_TYPE_STRING, "pyannet"),
        "general.name": (GGUF_TYPE_STRING, "pyannote-segmentation-3.0"),
        "general.alignment": (GGUF_TYPE_UINT32, GGUF_DEFAULT_ALIGNMENT),
        "pyannet.sample_rate": (GGUF_TYPE_UINT32, 16000),
        "pyannet.num_classes": (GGUF_TYPE_UINT32, 7),
        "pyannet.lstm_layers": (GGUF_TYPE_UINT32, 4),
        "pyannet.lstm_hidden": (GGUF_TYPE_UINT32, 128),
        "pyannet.sincnet_kernel_size": (GGUF_TYPE_UINT32, 251),
        "pyannet.sincnet_stride": (GGUF_TYPE_UINT32, 10),
    }

    tensor_infos = []
    current_offset = 0

    for name, data in tensors.items():
        ftype = choose_ftype(name, data)
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
    parser = argparse.ArgumentParser(description="Convert PyAnnote segmentation checkpoint to GGUF")
    parser.add_argument("model_path", help="Path to PyTorch checkpoint, usually segmentation/pytorch_model.bin")
    parser.add_argument("output_path", help="Path to output GGUF file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce logging")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet
    print(f"Loading segmentation checkpoint: {model_path}")
    state_dict = load_state_dict(str(model_path))
    print(f"Found {len(state_dict)} raw tensors")

    tensors = map_segmentation_tensors(state_dict, verbose=verbose)
    infos = write_gguf(args.output_path, tensors)

    f16_count = sum(1 for info in infos if info["ftype"] == GGML_TYPE_F16)
    f32_count = len(infos) - f16_count
    print(f"Wrote {args.output_path}")
    print(f"  tensors: {len(infos)}")
    print(f"  f16: {f16_count}")
    print(f"  f32: {f32_count}")


if __name__ == "__main__":
    main()
