import torch
import sys

model_path = "/home/aim/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/embedding/pytorch_model.bin"

try:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    print(f"Loaded embedding model from {model_path}")
    print("Keys and types:")
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {type(v)}")
            
    if 'hyper_parameters' in checkpoint:
        print("\nHyper-parameters:")
        print(checkpoint['hyper_parameters'])
        
except Exception as e:
    print(f"Error loading model: {e}")

