#!/usr/bin/env python3

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Inspect tensor groups inside a GGUF file")
    parser.add_argument("model_path", help="Path to GGUF file")
    args = parser.parse_args()

    import gguf

    model_path = args.model_path
    reader = gguf.GGUFReader(model_path)

    groups = {
        "segmentation": [],
        "embedding": [],
        "plda": [],
        "other": [],
    }

    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("sincnet.") or name.startswith("lstm.") or name.startswith("linear.") or name.startswith("classifier."):
            groups["segmentation"].append((name, tensor.shape))
        elif name.startswith("resnet."):
            groups["embedding"].append((name, tensor.shape))
        elif name.startswith("plda."):
            groups["plda"].append((name, tensor.shape))
        else:
            groups["other"].append((name, tensor.shape))

    print(f"Tensors in {model_path}:")
    for group_name, tensors in groups.items():
        print(f"\n[{group_name}] count={len(tensors)}")
        for name, shape in tensors:
            print(f"  {name} | shape: {shape}")


if __name__ == "__main__":
    main()
