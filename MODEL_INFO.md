# Trained Model Information

trained_model.pth
Target Accuracy: 85.82%

Best Configuration: SiLU + Adam + LR=0.001

From comprehensive experiments

Usage
python
import torch
model = VGG6(activation='silu')
model.load_state_dict(torch.load('trained_model.pth'))
