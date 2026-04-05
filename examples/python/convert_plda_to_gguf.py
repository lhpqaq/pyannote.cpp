#!/usr/bin/env python3
"""Convert PLDA parameters to GGUF format for the current C++ pipeline."""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

XVEC_DIM = 256
LDA_DIM = 128

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGML_TYPE_F64 = 28
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8

PLDA_BIN_MAGIC = b"PLDA"
PLDA_BIN_VERSION = 1


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


def load_from_npz(transform_path: str, plda_path: str):
    from scipy.linalg import eigh

    x = np.load(transform_path)
    mean1, mean2, lda = x["mean1"], x["mean2"], x["lda"]

    p = np.load(plda_path)
    plda_mu, plda_tr, plda_psi = p["mu"], p["tr"], p["psi"]

    w_mat = np.linalg.inv(plda_tr.T.dot(plda_tr))
    b_mat = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(b_mat, w_mat)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    return (
        mean1.astype(np.float64),
        mean2.astype(np.float64),
        lda.astype(np.float64),
        plda_mu.astype(np.float64),
        plda_tr.astype(np.float64),
        plda_psi.astype(np.float64),
    )


def load_from_bin(bin_path: str):
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != PLDA_BIN_MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != PLDA_BIN_VERSION:
            raise ValueError(f"bad version: {version}")

        mean1 = np.frombuffer(f.read(XVEC_DIM * 8), dtype=np.float64).copy()
        mean2 = np.frombuffer(f.read(LDA_DIM * 8), dtype=np.float64).copy()
        lda = np.frombuffer(f.read(XVEC_DIM * LDA_DIM * 8), dtype=np.float64).copy().reshape(XVEC_DIM, LDA_DIM)
        plda_mu = np.frombuffer(f.read(LDA_DIM * 8), dtype=np.float64).copy()
        plda_tr = np.frombuffer(f.read(LDA_DIM * LDA_DIM * 8), dtype=np.float64).copy().reshape(LDA_DIM, LDA_DIM)
        plda_psi = np.frombuffer(f.read(LDA_DIM * 8), dtype=np.float64).copy()
        trailing = f.read()
        if trailing:
            raise ValueError("unexpected trailing bytes in plda.bin")

    return mean1, mean2, lda, plda_mu, plda_tr, plda_psi


def write_plda_gguf(output_path: str, mean1, mean2, lda, plda_mu, plda_tr, plda_psi):
    metadata = {
        "general.architecture": (GGUF_TYPE_STRING, "plda"),
        "general.name": (GGUF_TYPE_STRING, "pyannote-plda-vbx"),
        "general.alignment": (GGUF_TYPE_UINT32, GGUF_DEFAULT_ALIGNMENT),
        "plda.xvec_dim": (GGUF_TYPE_UINT32, XVEC_DIM),
        "plda.lda_dim": (GGUF_TYPE_UINT32, LDA_DIM),
    }

    tensors = [
        ("plda.mean1", np.ascontiguousarray(mean1, dtype=np.float64)),
        ("plda.mean2", np.ascontiguousarray(mean2, dtype=np.float64)),
        ("plda.lda", np.ascontiguousarray(lda, dtype=np.float64)),
        ("plda.mu", np.ascontiguousarray(plda_mu, dtype=np.float64)),
        ("plda.tr", np.ascontiguousarray(plda_tr, dtype=np.float64)),
        ("plda.psi", np.ascontiguousarray(plda_psi, dtype=np.float64)),
    ]

    tensor_infos = []
    current_offset = 0
    for name, data in tensors:
        tensor_infos.append(
            {
                "name": name,
                "data": data,
                "ftype": GGML_TYPE_F64,
                "offset": current_offset,
            }
        )
        current_offset += align_offset(data.nbytes)

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


def main():
    parser = argparse.ArgumentParser(description="Convert PLDA parameters to GGUF")
    parser.add_argument("--from-bin", help="Existing plda.bin file")
    parser.add_argument("--transform-npz", help="Path to xvec_transform.npz")
    parser.add_argument("--plda-npz", help="Path to plda.npz")
    parser.add_argument("-o", "--output", default="plda.gguf", help="Output GGUF path")
    args = parser.parse_args()

    if args.from_bin:
        tensors = load_from_bin(args.from_bin)
    elif args.transform_npz and args.plda_npz:
        tensors = load_from_npz(args.transform_npz, args.plda_npz)
    else:
        parser.error("provide either --from-bin or both --transform-npz and --plda-npz")

    output_path = Path(args.output)
    write_plda_gguf(str(output_path), *tensors)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
