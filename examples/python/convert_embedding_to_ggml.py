import torch
import sys
import numpy as np
import gguf
import os
from pathlib import Path
from scipy.linalg import eigh

def fold_batch_norm(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var):
    # conv_w: (out_channels, in_channels, kH, kW)
    # bn params: (out_channels)
    
    eps = 1e-5
    scale = bn_w / torch.sqrt(bn_var + eps)
    shift = bn_b - bn_mean * scale
    
    # Reshape scale and shift for broadcasting
    # For Conv2d weights: (out, in, k, k) -> scale needs to be (out, 1, 1, 1)
    scale_w = scale.view(-1, 1, 1, 1)
    
    new_conv_w = conv_w * scale_w
    
    if conv_b is None:
        new_conv_b = shift
    else:
        new_conv_b = conv_b * scale + shift
        
    return new_conv_w, new_conv_b

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_embedding_to_ggml.py <path_to_pytorch_model.bin> <output.gguf>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    gguf_writer = gguf.GGUFWriter(output_path, "pyannote-embedding")

    # Hyperparameters
    # Input features: fbank 80 dim
    gguf_writer.add_int32("fbank.n_mels", 80)
    gguf_writer.add_int32("embedding.dim", 256) # Based on inspection, check seg_1 weight shape later

    print("Converting weights...")
    
    # We need to iterate layers and fold BatchNorms.
    # The structure is:
    # resnet.conv1 (Conv2d)
    # resnet.bn1 (BatchNorm2d)
    # resnet.layerX.Y.conv1 (Conv2d)
    # resnet.layerX.Y.bn1 (BatchNorm2d)
    # ...
    
    # Helper to find bn params for a conv name
    # e.g. conv_name = "resnet.conv1.weight" -> bn_prefix = "resnet.bn1"
    # e.g. conv_name = "resnet.layer1.0.conv1.weight" -> bn_prefix = "resnet.layer1.0.bn1"
    
    def get_bn_prefix(conv_name):
        parts = conv_name.split(".")
        if parts[-1] == "weight":
            parts = parts[:-1] # remove weight
        
        # Replace convX with bnX
        if parts[-1].startswith("conv"):
            parts[-1] = parts[-1].replace("conv", "bn")
            return ".".join(parts)
        return None

    processed_keys = set()
    
    for k in state_dict.keys():
        if k in processed_keys:
            continue
            
        if "num_batches_tracked" in k:
            continue
            
        v = state_dict[k]
        
        # Check if it's a Conv weight that has a corresponding BN
        if "conv" in k and "weight" in k and v.dim() == 4:
            bn_prefix = get_bn_prefix(k)
            if bn_prefix and (bn_prefix + ".weight") in state_dict:
                print(f"Folding BN {bn_prefix} into {k}")
                
                conv_w = v
                conv_b = state_dict.get(k.replace("weight", "bias")) # Conv usually has no bias in ResNet
                
                bn_w = state_dict[bn_prefix + ".weight"]
                bn_b = state_dict[bn_prefix + ".bias"]
                bn_mean = state_dict[bn_prefix + ".running_mean"]
                bn_var = state_dict[bn_prefix + ".running_var"]
                
                new_w, new_b = fold_batch_norm(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var)
                
                # Add to GGUF
                # Store weights as F32
                gguf_writer.add_tensor(k, new_w.numpy().astype(np.float32))
                gguf_writer.add_tensor(k.replace("weight", "bias"), new_b.numpy().astype(np.float32))
                
                # Mark BN keys as processed
                processed_keys.add(bn_prefix + ".weight")
                processed_keys.add(bn_prefix + ".bias")
                processed_keys.add(bn_prefix + ".running_mean")
                processed_keys.add(bn_prefix + ".running_var")
                processed_keys.add(k)
                if conv_b is not None:
                    processed_keys.add(k.replace("weight", "bias"))
                continue
        
        # Downsample/Shortcut layers often have conv+bn too
        # resnet.layer2.0.shortcut.0 (Conv)
        # resnet.layer2.0.shortcut.1 (BN)
        if "shortcut.0.weight" in k:
            bn_prefix = k.replace("shortcut.0.weight", "shortcut.1")
            if (bn_prefix + ".weight") in state_dict:
                print(f"Folding BN {bn_prefix} into {k}")
                
                conv_w = v
                conv_b = state_dict.get(k.replace("weight", "bias"))
                
                bn_w = state_dict[bn_prefix + ".weight"]
                bn_b = state_dict[bn_prefix + ".bias"]
                bn_mean = state_dict[bn_prefix + ".running_mean"]
                bn_var = state_dict[bn_prefix + ".running_var"]
                
                new_w, new_b = fold_batch_norm(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var)
                
                gguf_writer.add_tensor(k, new_w.numpy().astype(np.float32))
                gguf_writer.add_tensor(k.replace("weight", "bias"), new_b.numpy().astype(np.float32))
                
                processed_keys.add(bn_prefix + ".weight")
                processed_keys.add(bn_prefix + ".bias")
                processed_keys.add(bn_prefix + ".running_mean")
                processed_keys.add(bn_prefix + ".running_var")
                processed_keys.add(k)
                continue

        # Linear layers (seg_1, seg_2)
        # resnet.seg_1.weight
        # resnet.seg_1.bias
        if "seg_" in k and "weight" in k:
            # Check for seg_bn_1
            # In ResNet init: if two_emb_layer: seg_1 -> relu -> seg_bn_1 -> seg_2
            # seg_bn_1 is BatchNorm1d(embed_dim, affine=False). 
            
            print(f"Adding {k}, shape {v.shape}, type F32")
            gguf_writer.add_tensor(k, v.numpy().astype(np.float32))
            processed_keys.add(k)
            continue
            
        if "seg_" in k and "bias" in k:
            print(f"Adding {k}, shape {v.shape}, type F32")
            gguf_writer.add_tensor(k, v.numpy().astype(np.float32))
            processed_keys.add(k)
            continue

        # Any other weights not processed
        if k not in processed_keys:
            # Ignore BN params that might have been skipped if logic wasn't perfect (safe fallback?)
            # No, we should only add what we need.
            # print(f"Skipping {k}")
            pass

    # --- VBx / PLDA Model integration ---
    print("Checking for PLDA model files...")
    # Assuming model_path is like ".../embedding/pytorch_model.bin"
    # We look for ".../plda/xvec_transform.npz" and ".../plda/plda.npz"
    
    model_dir = Path(model_path).parent
    snapshot_dir = model_dir.parent
    plda_dir = snapshot_dir / "plda"
    
    path_to_transform = plda_dir / "xvec_transform.npz"
    path_to_plda = plda_dir / "plda.npz"
    
    if path_to_transform.exists() and path_to_plda.exists():
        print(f"Found PLDA files at {plda_dir}")
        try:
            x = np.load(path_to_transform)
            mean1, mean2, lda = x["mean1"], x["mean2"], x["lda"]

            p = np.load(path_to_plda)
            plda_mu, plda_tr, plda_psi = p["mu"], p["tr"], p["psi"]
            
            # Compute final VBx matrices
            # within-class, between-class matrices (W, B)
            W_mat = np.linalg.inv(plda_tr.T.dot(plda_tr))
            B_mat = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))

            # Solve generalized eigenvalue problem
            acvar, wccn = eigh(B_mat, W_mat)
            plda_psi_final = acvar[::-1]
            plda_tr_final = wccn.T[::-1]
            
            print("Adding VBx/PLDA tensors to GGUF...")
            gguf_writer.add_tensor("vbx.mean1", mean1.astype(np.float32))
            gguf_writer.add_tensor("vbx.mean2", mean2.astype(np.float32))
            gguf_writer.add_tensor("vbx.lda", lda.astype(np.float32))
            gguf_writer.add_tensor("vbx.plda_mu", plda_mu.astype(np.float32))
            gguf_writer.add_tensor("vbx.plda_tr", plda_tr_final.astype(np.float32))
            gguf_writer.add_tensor("vbx.plda_psi", plda_psi_final.astype(np.float32))
            
        except Exception as e:
            print(f"Error processing PLDA files: {e}")
    else:
        print(f"PLDA files not found in {plda_dir}. Skipping VBx integration.")


    print("Writing GGUF file...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()
