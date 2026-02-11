# Complete MNIST Classification Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [The MNIST Dataset](#the-mnist-dataset)
4. [Complete Pipeline Breakdown](#complete-pipeline-breakdown)
5. [Model Architecture](#model-architecture)
6. [Training Process](#training-process)
7. [Evaluation & Results](#evaluation--results)
8. [Running the Code](#running-the-code)

---

## Project Overview

This project demonstrates a **Convolutional Neural Network (CNN)** built with PyTorch to classify handwritten digits from the MNIST (Modified National Institute of Standards and Technology) dataset. The goal is to train a model that can accurately predict which digit (0-9) is shown in a 28×28 pixel grayscale image.

### Key Technologies
- **PyTorch**: Deep learning framework for building neural networks
- **Torchvision**: Computer vision utilities (datasets, transforms)
- **GPU/MPS Support**: Code automatically uses GPU (CUDA), Apple Metal Performance Shaders (MPS), or CPU

---

## Environment Setup

### Dependencies

The project requires the following packages:

```
torch              # PyTorch deep learning framework
torchvision        # Computer vision datasets and transforms
matplotlib         # Visualization library
jupyter            # Interactive notebooks
numpy              # Numerical computing
```

### Installation Steps

#### Option 1: Using Conda (Recommended)
```bash
conda env create -f environment.yaml
conda activate pytorch_env
```

The `environment.yaml` file specifies:
- **Python 3.11** (stable version compatible with PyTorch)
- **PyTorch** from the official PyTorch conda channel
- **Torchvision** (includes MNIST dataset loader)
- **Jupyter** and **IPython kernel** (for notebook support)
- Additional utilities (matplotlib, numpy)

#### Option 2: Using pip
```bash
pip install torch torchvision matplotlib jupyter numpy
```

### Verifying Installation

Check that PyTorch is installed correctly:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Check GPU availability
```

---

## The MNIST Dataset

### What is MNIST?

MNIST is a foundational dataset in machine learning containing:
- **60,000 training images** for model training
- **10,000 test images** for evaluation
- **28×28 pixel** grayscale images
- **10 classes** (digits 0-9)
- **Highly labeled and pre-processed** (ready to use)

### Why MNIST?

- Simple enough to train quickly (good for learning)
- Complex enough to demonstrate real neural networks
- Standardized benchmark for comparing models
- Well-documented with high baseline results

### Data Characteristics

**Pixel Values**: 0-255 (original), normalized to 0-1 (after ToTensor)

**Normalization Parameters** (used in this project):
- **Mean**: 0.1307
- **Standard Deviation**: 0.3081

These values are the actual mean and std of the MNIST dataset, computed across all training images.

### Why Normalize Data?

Normalization brings data to a standard range (mean=0, std=1), which helps the neural network:
- Converge faster during training
- Avoid numerical instability
- Reduce internal covariate shift

---

## Complete Pipeline Breakdown

### Step 1: Import Libraries & Check Device

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
```

**What this does:**
- Imports all necessary libraries
- **Device Selection Logic**:
  1. Check if CUDA GPU is available (NVIDIA)
  2. Fallback to MPS (Apple Silicon/Metal)
  3. Fallback to CPU if neither available
- This prioritizes fastest available compute

**Device Types:**
- **CUDA**: NVIDIA GPUs (10-100× faster than CPU)
- **MPS**: Apple Metal Performance Shaders (M1/M2/M3 chips)
- **CPU**: Fallback option (slowest but always available)

### Step 2: Set Hyperparameters

```python
BATCH_SIZE = 64          # Number of samples per batch
LEARNING_RATE = 0.001    # Step size for optimizer
EPOCHS = 5               # Number of full passes through dataset
```

**What Each Parameter Does:**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **BATCH_SIZE** | 64 | Process 64 images at a time. Larger = faster training but more memory. Smaller = more stable gradients but slower. |
| **LEARNING_RATE** | 0.001 | Step size for weight updates. Too large = divergence. Too small = very slow convergence. |
| **EPOCHS** | 5 | Total training rounds through entire dataset. More epochs = longer training, potential overfitting. |

### Step 3: Data Transformation & Loading

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

**Data Transformation Pipeline:**

1. **ToTensor()**: 
   - Converts PIL Image to PyTorch tensor
   - Converts pixel values from [0-255] to [0.0-1.0]
   - Changes shape from (H, W, C) to (C, H, W)

2. **Normalize()**: 
   - Applies: `(x - mean) / std`
   - Centers data around 0 with std of 1
   - Improves neural network training stability

**DataLoader Explained:**

- **DataLoader** is a wrapper that:
  - Batches data into groups of 64 images
  - Handles memory-efficient iteration
  - Allows parallel data loading
  - **shuffle=True** for training (randomizes order, prevents overfitting)
  - **shuffle=False** for testing (keeps consistent order for evaluation)

**Dataset Sizes:**
- Training: 60,000 images
- Testing: 10,000 images

---

## Model Architecture

### CNN Architecture Overview

```
Input (1, 28, 28)
    ↓
Conv Layer 1: 1→16 filters (3×3 kernel)
    ↓
ReLU Activation
    ↓
Max Pool: 2×2 stride
    ↓
Conv Layer 2: 16→32 filters (3×3 kernel)
    ↓
ReLU Activation
    ↓
Max Pool: 2×2 stride
    ↓
Flatten to 1D: 32 * 7 * 7 = 1,568 features
    ↓
Fully Connected Layer 1: 1,568→128 neurons
    ↓
ReLU Activation
    ↓
Fully Connected Layer 2: 128→10 outputs
    ↓
Output (10 classes: digits 0-9)
```

### Detailed Layer Breakdown

#### 1. Convolutional Layer 1
```python
self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
```

- **Input**: 1 channel (grayscale)
- **Output**: 16 channels (learned features)
- **Kernel Size**: 3×3 filters
- **Padding**: 1 (preserves spatial dimensions)
- **What it does**: Detects low-level features (edges, textures)
- **Output shape**: (16, 28, 28)

#### 2. ReLU Activation
```python
self.relu = nn.ReLU()
```

- **Function**: `ReLU(x) = max(0, x)`
- **Purpose**: Introduces non-linearity (enables learning complex patterns)
- **Advantage**: Computationally efficient, prevents vanishing gradient problem

#### 3. Max Pooling Layer 1
```python
self.pool = nn.MaxPool2d(2, 2)
```

- **Kernel Size**: 2×2
- **Stride**: 2 (moves by 2 pixels)
- **What it does**: 
  - Reduces spatial dimensions by 50%
  - Keeps maximum value in each 2×2 region
  - Makes features translation-invariant
- **Output after pool**: (16, 14, 14)

#### 4. Convolutional Layer 2
```python
self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
```

- **Input**: 16 channels (from previous conv)
- **Output**: 32 channels (more abstract features)
- **Kernel Size**: 3×3 filters
- **Purpose**: Detects higher-level features (shapes, patterns)
- **Output shape**: (32, 14, 14)

#### 5. Max Pooling Layer 2
- Reduces dimensions again: (32, 7, 7)

#### 6. Flatten
```python
x = x.view(-1, 32 * 7 * 7)
```

- **Converts**: (batch_size, 32, 7, 7) → (batch_size, 1568)
- **-1**: Automatically calculates batch dimension
- **Purpose**: Prepare for fully connected layers

#### 7. Fully Connected Layer 1
```python
self.fc1 = nn.Linear(1568, 128)
```

- **Input**: 1,568 flattened features
- **Output**: 128 neurons
- **Purpose**: Learn complex decision boundaries
- **Total parameters**: 1,568 × 128 = ~200K weights

#### 8. Fully Connected Layer 2 (Output)
```python
self.fc2 = nn.Linear(128, 10)
```

- **Input**: 128 neurons
- **Output**: 10 neurons (one per digit class)
- **Purpose**: Final classification logits
- **Total parameters**: 128 × 10 = 1,280 weights

### Total Parameters

- **Trainable parameters**: ~220,000
- Relatively small model (suitable for CPU/MPS)
- Fast training and inference

---

## Training Process

### Loss Function & Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

#### Cross Entropy Loss
- **Function**: Measures difference between predicted and true probability distributions
- **Why**: Standard loss for multi-class classification
- **Formula**: 
  ```
  Loss = -Σ(true_label * log(predicted_probability))
  ```
- **Output**: Higher loss = worse predictions

#### Adam Optimizer
- **Adaptive Moment Estimation**
- **Advantages**:
  - Adapts learning rate per parameter
  - Faster convergence than vanilla SGD
  - Less sensitive to learning rate tuning
- **How it works**:
  - Maintains momentum (first moment) and adaptive rates (second moment)
  - Updates: `θ = θ - α * m / (√v + ε)`

### Training Function Explained

```python
def train(model, device, train_loader, optimizer, epoch):
    model.train()                              # Set to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move to GPU/MPS/CPU
        optimizer.zero_grad()                  # Clear old gradients
        output = model(data)                   # Forward pass
        loss = criterion(output, target)       # Calculate loss
        loss.backward()                        # Compute gradients (backpropagation)
        optimizer.step()                       # Update weights
        if batch_idx % 100 == 0:
            print(f'Loss: {loss.item():.6f}')  # Print progress every 100 batches
```

#### Step-by-Step Breakdown

1. **model.train()**: 
   - Sets model to training mode
   - Enables dropout and batch normalization variations

2. **Move Data to Device**: 
   - Transfers tensors to GPU/MPS (if available)
   - Ensures all computations happen on same device

3. **optimizer.zero_grad()**: 
   - Clears gradients from previous iteration
   - Essential to prevent gradient accumulation

4. **Forward Pass (model(data))**: 
   - Inputs: batch of images (64, 1, 28, 28)
   - Passes through all layers
   - Outputs: class logits (64, 10)

5. **Calculate Loss**: 
   - Compares predictions with true labels
   - Single scalar value

6. **Backward Pass (loss.backward())**: 
   - Computes gradients via backpropagation
   - Uses chain rule through all layers
   - Stores gradients in each parameter

7. **Optimizer Step (optimizer.step())**: 
   - Updates weights based on gradients
   - Direction: opposite to gradient (downhill)
   - Magnitude: controlled by learning rate

#### What Happens Each Epoch

- **1 epoch** = one complete pass through all 60,000 training images
- **Number of batches**: 60,000 / 64 ≈ 938 batches per epoch
- **Total iterations**: 5 epochs × 938 ≈ 4,690 weight updates
- **Training time**: ~30-60 seconds on GPU, ~5 minutes on CPU

---

## Evaluation & Results

### Testing Function

```python
def test(model, device, test_loader):
    model.eval()                              # Set to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():                     # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # Get max prediction
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Accuracy: {accuracy:.1f}%')
```

#### Key Concepts

1. **model.eval()**: 
   - Sets model to evaluation mode
   - Disables dropout/batch norm randomness
   - Makes predictions deterministic

2. **torch.no_grad()**: 
   - Disables automatic differentiation
   - Saves memory and computation time
   - Appropriate since we don't need gradients for testing

3. **argmax(dim=1)**: 
   - Finds the highest probability class per sample
   - Returns class index (0-9)

4. **Accuracy Calculation**: 
   - Compares predictions to ground truth
   - Percentage of correct predictions

### Expected Results

After 5 epochs, you should see:
- **Training loss**: Decreases from ~0.5 to ~0.01
- **Test accuracy**: Increases from ~90% to ~98%
- **Improvement curve**: Steepest improvement in first 2 epochs

### Training Loop

```python
for epoch in range(1, EPOCHS + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

**What happens**:
- Each epoch: train on all training data, then evaluate on test data
- Epochs 1-2: Rapid accuracy improvement (large loss decrease)
- Epochs 3-5: Diminishing returns (smaller improvements)
- Potential: Overfitting after epoch 5 (would see training accuracy continue rising but test accuracy plateauing)

---

## Running the Code

### Prerequisites

1. **Activate environment**:
   ```bash
   conda activate pytorch_env
   ```

2. **Verify PyTorch installation**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

### Running the Notebook

#### Option 1: Jupyter Notebook (Interactive)
```bash
cd /Users/sean/Documents/gpu
jupyter notebook mnist_classifier.ipynb
```

Then:
- Open the notebook in browser
- Run cells sequentially (Shift+Enter)
- See outputs and visualizations immediately

#### Option 2: Jupyter Lab (Modern Interface)
```bash
jupyter lab mnist_classifier.ipynb
```

#### Option 3: VS Code with Jupyter Extension
1. Install "Jupyter" extension in VS Code
2. Open `mnist_classifier.ipynb`
3. Run cells directly in editor

### What Happens When You Run

**Cell 1**: Initializes libraries and device detection
- Output: `Using device: [cuda/mps/cpu]`

**Cell 2**: Sets hyperparameters (no output)

**Cell 3**: Downloads MNIST dataset and creates DataLoaders
- First run: Downloads ~11 MB of data (one-time)
- Output: `Training samples: 60000` / `Test samples: 10000`
- Creates `data/MNIST/` folder

**Cell 4**: Displays sample images
- Output: 10 random training images displayed
- Visual check that data loaded correctly

**Cell 5**: Prints model architecture
- Output: All layers with dimensions
- Shows total parameters

**Cell 6**: Sets up loss and optimizer (no visible output)

**Cell 7**: Training loop
- Output: Loss values every 100 batches × 5 epochs
- Watch loss decrease as training progresses
- Duration: 30s-5min depending on device

**Cell 8**: Evaluation function definition (no output)

**Cell 9**: Main training loop
- Runs training and testing for each epoch
- Output: Test accuracy after each epoch
- Total accuracy should improve from ~90% to ~98%

**Cell 10**: Visualization of predictions
- Output: 10 test images with predicted and true labels
- Inspect if model is making reasonable errors

### Monitoring Training

Look for these signs of healthy training:
- ✅ Loss decreases consistently
- ✅ Test accuracy increases consistently
- ✅ Loss values make sense (0.01-0.5 range)
- ✅ No NaN or infinity values

### Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| **Out of Memory** | Batch too large for device | Reduce BATCH_SIZE to 32 or 16 |
| **Very slow training** | Using CPU | Ensure `device: mps` shows in output |
| **Loss stays high** | Learning rate too low | Increase LEARNING_RATE to 0.01 |
| **Loss becomes NaN** | Learning rate too high | Decrease LEARNING_RATE to 0.0001 |
| **Dataset not found** | Network issues | Manually download from yann.lecun.com |

---

## Key Learning Outcomes

After running this project, you'll understand:

1. **Data Pipeline**: Loading, transforming, batching data
2. **CNN Architecture**: How convolutional and pooling layers work
3. **Training Loop**: Forward pass, loss calculation, backpropagation, optimization
4. **Hyperparameters**: Impact of batch size, learning rate, epochs
5. **GPU Acceleration**: Device selection and tensor movement
6. **Model Evaluation**: Accuracy metrics and performance analysis
7. **PyTorch Basics**: Tensors, models, optimizers, data loaders

---

## Further Exploration

### Experiments to Try

1. **Increase EPOCHS to 10**: See how accuracy plateaus
2. **Reduce BATCH_SIZE to 32**: Observe different training dynamics
3. **Change LEARNING_RATE**: Find optimal value
4. **Add more Conv layers**: See how depth affects performance
5. **Use different optimizer**: Try SGD or RMSprop instead of Adam
6. **Add dropout**: Prevent overfitting on higher epoch counts
7. **Save model weights**: Use `torch.save(model.state_dict(), 'model.pth')`
8. **Load saved model**: Use `model.load_state_dict(torch.load('model.pth'))`

### Next Steps

- **CIFAR-10**: Color images, more classes (requires deeper network)
- **ImageNet Transfer Learning**: Use pre-trained models
- **Custom Dataset**: Apply CNN to your own images
- **Model Deployment**: Save and load for inference in production

---

## Summary

This MNIST classifier demonstrates the complete deep learning pipeline:
- Data preparation and augmentation
- Neural network architecture design
- Training with backpropagation
- Model evaluation and metrics
- GPU acceleration

With just ~1,000 lines of code, you've built a working classifier achieving ~98% accuracy on handwritten digit recognition!

