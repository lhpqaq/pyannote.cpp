import gguf
import sys

if len(sys.argv) < 2:
    print("Usage: python check_gguf_tensors.py <model.gguf>")
    sys.exit(1)

model_path = sys.argv[1]
reader = gguf.GGUFReader(model_path)

print(f"Tensors in {model_path}:")
found_vbx = False
for tensor in reader.tensors:
    if tensor.name.startswith("vbx."):
        print(f"  {tensor.name} | shape: {tensor.shape}")
        found_vbx = True

if found_vbx:
    print("\nVBx tensors FOUND.")
else:
    print("\nVBx tensors NOT FOUND.")

