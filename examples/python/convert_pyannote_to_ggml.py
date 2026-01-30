import torch
import sys
import numpy as np
import gguf
from asteroid_filterbanks import ParamSincFB

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_pyannote_to_ggml.py <path_to_pytorch_model.bin> <output.gguf>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    gguf_writer = gguf.GGUFWriter(output_path, "pyannote-segmentation")

    # Hyperparameters from inspection
    # SincNet
    stride = 10
    sample_rate = 16000
    gguf_writer.add_int32("sincnet.stride", stride)
    gguf_writer.add_int32("sincnet.sample_rate", sample_rate)
    
    # LSTM
    gguf_writer.add_int32("lstm.hidden_size", 128)
    gguf_writer.add_int32("lstm.num_layers", 4)
    gguf_writer.add_bool("lstm.bidirectional", True)
    
    # Linear
    gguf_writer.add_int32("linear.hidden_size", 128)
    gguf_writer.add_int32("linear.num_layers", 2) 

    # 1. SincNet First Layer Conversion
    print("Generating SincNet filters...")
    
    # Extract filterbank state
    fb_prefix = "sincnet.conv1d.0.filterbank."
    fb_state = {}
    for k, v in state_dict.items():
        if k.startswith(fb_prefix):
            fb_state[k[len(fb_prefix):]] = v
    
    if not fb_state:
        print("Error: Could not find SincNet filterbank parameters in state_dict")
        sys.exit(1)

    # Initialize ParamSincFB
    # Note: These parameters must match the model's configuration
    filterbank = ParamSincFB(
        n_filters=80,
        kernel_size=251,
        stride=stride,
        sample_rate=sample_rate,
        min_low_hz=50,
        min_band_hz=50
    )
    
    # Load state
    # ParamSincFB might have buffers like window_ and n_ which are important
    missing, unexpected = filterbank.load_state_dict(fb_state, strict=True)
    if missing:
        print(f"Warning: Missing keys for filterbank: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys for filterbank: {unexpected}")
        
    # Get filters
    # The 'filters' property computes the filters based on current parameters
    if callable(filterbank.filters):
        filters = filterbank.filters()
    else:
        filters = filterbank.filters
    
    print(f"Generated SincNet filters shape: {filters.shape}") # Should be (80, 1, 251)
    
    # Add SincNet filters to GGUF
    # We rename it to what we expect in C++: sincnet.conv1d.0.weight
    filters_np = filters.detach().numpy().astype(np.float16)
    print(f"Adding sincnet.conv1d.0.weight, shape {filters_np.shape}, type {filters_np.dtype}")
    gguf_writer.add_tensor("sincnet.conv1d.0.weight", filters_np)
    
    # 2. Process all other weights
    for k, v in state_dict.items():
        # Skip filterbank params as we handled them
        if k.startswith(fb_prefix):
            continue
            
        data = v.numpy()
        
        # Decide type based on name
        # Weights for conv/linear/lstm usually F16 for efficiency
        # Biases and Norm params usually F32 for precision and compatibility with binary ops
        if "weight" in k and "norm" not in k:
            data = data.astype(np.float16)
        else:
            data = data.astype(np.float32)
            
        print(f"Adding {k}, shape {data.shape}, type {data.dtype}")
        gguf_writer.add_tensor(k, data)

    print("Writing GGUF file...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()
