# vgg6-cifar10-DL-assignment

# VGG6 CIFAR-10 Classification - Deep Learning Assignment

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning project implementing VGG6 architecture on CIFAR-10 dataset with extensive hyperparameter optimization, achieving **85.82% test accuracy**.

##  Key Features

- **VGG6 Architecture**: Custom lightweight VGG-style CNN with batch normalization
- **Comprehensive Experiments**: Tested 6 activation functions and 6 optimizers
- **Hyperparameter Optimization**: Systematic search for optimal learning rates, batch sizes
- **Reproducible Results**: Complete configuration files and trained models
- **Well-Documented**: Detailed experiment summaries and model information

##  Results Summary

| Model | Activation | Optimizer | Learning Rate | Batch Size | Accuracy |
|-------|------------|-----------|---------------|------------|----------|
| **Best Model** | SiLU | Adam | 0.001 | 64 | **85.82%** |
| Baseline | ReLU | Adam | 0.001 | 128 | 85.14% |

### Activation Functions Tested
- ReLU, SiLU, LeakyReLU, GELU, Mish, Tanh

### Optimizers Tested  
- Adam, SGD, RMSprop, Adagrad, Adadelta, AdamW

## 🏗️ Project Structure
├── train.py # Main training script
├── requirements.txt # Dependencies
├── trained_model.pth # Best trained model weights
├── best_config.json # Best hyperparameters
├── baseline_config.json # Baseline configuration
├── experiment_summary.json # Experiment results
├── config.json # Training configuration
└── README.md # Project documentation

##  Installation & Usage
### Prerequisites
```bash
pip install -r requirements.txt

# Quick Start
bash
# Train the model with best configuration
python train.py

# Load Pre-trained Model
python
import torch
from train import VGG6

# Load best model
model = VGG6(activation='silu')
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

📈 Model Architecture
VGG6 Network:

text
Input (3, 32, 32)
├── Conv2d(3→64) → BatchNorm → Activation
├── Conv2d(64→64) → BatchNorm → Activation → MaxPool2d
├── Conv2d(64→128) → BatchNorm → Activation
├── Conv2d(128→128) → BatchNorm → Activation → MaxPool2d
└── AdaptiveAvgPool2d → Flatten → Linear(128, 10)
🔬 Experiment Details
Dataset: CIFAR-10 (50,000 training, 10,000 test images)

Image Size: 32×32 RGB images, 10 classes

Total Configurations: 16 combinations tested

Training: 30 epochs per configuration

Validation: Standard train-test split

📁 File Descriptions
train.py: Complete training pipeline with data loading, model definition, and training loop

best_config.json: Optimal hyperparameters achieving 85.82% accuracy

baseline_config.json: Standard ReLU+Adam baseline configuration

experiment_summary.json: Summary of all experiments and findings

trained_model.pth: Pre-trained weights of the best performing model

requirements.txt: Dependencies required to run the project

🎯 Key Findings
SiLU activation outperformed traditional ReLU by ~0.7%

Adam optimizer consistently delivered best results across activations

Smaller batch sizes (64) worked better than larger ones (128)

Learning rate 0.001 provided optimal convergence speed and stability

🧪 Model Usage Guide
Loading the Pre-trained Model
python
import torch
from train import VGG6
from torchvision import transforms
from PIL import Image

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG6(activation='silu')
model.load_state_dict(torch.load('trained_model.pth'))
model.to(device)
model.eval()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Prediction function
def predict(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

# Example usage
# predicted_class, confidence = predict(your_image)
# print(f"Predicted: {class_names[predicted_class]} (confidence: {confidence:.2f})")
📊 Performance Metrics
Best Accuracy: 85.82%

Training Time: ~30 minutes on GPU

Inference Speed: ~1000 images/second on GPU

Model Size: 1.2 MB

Parameters: ~300,000

**Verify model size:**
```bash
ls -lh trained_model.pth

🤝 Contributing
Feel free to fork this project and experiment with:

Different network architectures

Additional activation functions

Advanced optimization techniques

Data augmentation strategies

📄 License
This project is licensed under the MIT License.

🙏 Acknowledgments
CIFAR-10 dataset providers

PyTorch development team

Deep learning research community

⭐ If you find this project useful, please give it a star!
