# overhead_object_projector
This script is used to simulate and train overhead projection from images.

## Installation
```
conda create -n overhead_object_projector python=3.8
conda activate overhead_object_projector
pip install -r requirements.txt
```

Please install torch, torchvision by considering the compatible CUDA version. Tested with:
torch-1.12.1
torchvision-0.13.1

```
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

Note: The W&B sweeper produces configurations by considering the ranges and distributions of the hyperparameters and prints 
them to temp_params.py before the training with the decided configuration.
