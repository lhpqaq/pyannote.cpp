import torch
import sys
import yaml

# Add the project path if necessary, though we are relying on installed packages
# sys.path.append("/home/aim/project/pyannote-audio")

try:
    from pyannote.audio.models.segmentation import PyanNet
    from pyannote.audio.core.task import Specifications, Problem, Resolution

    # Define dummy specifications as seen in the checkpoint
    specifications = Specifications(
        problem=Problem.MONO_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=10.0,
        classes=['speaker#1', 'speaker#2', 'speaker#3']
    )

    # These params come from the printed hyper_parameters in the previous step
    hparams = {
        'sample_rate': 16000, 
        'num_channels': 1, 
        'sincnet': {'stride': 10, 'sample_rate': 16000}, 
        'lstm': {'hidden_size': 128, 'num_layers': 4, 'bidirectional': True, 'monolithic': True, 'dropout': 0.5, 'batch_first': True}, 
        'linear': {'hidden_size': 128, 'num_layers': 2}
    }

    model = PyanNet(specifications=specifications, **hparams)
    print("Model Architecture:")
    print(model)

except ImportError as e:
    print(f"Could not import pyannote.audio: {e}")
    # Fallback or just exit
    pass
except Exception as e:
    print(f"Error instantiating model: {e}")

