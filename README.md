<div align="center">

# 🧠 ASL-Sign-Language-Classifier

### **Advanced Real-Time American Sign Language Recognition Pipeline Using PyTorch**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-000000?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9AB00.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>A state-of-the-art Deep Learning project designed to classify static American Sign Language (ASL) gestures ($A \text{ to } Z$, excluding $J/Z$) from raw pixel matrices. This pipeline integrates custom structural Convolutional Blocks, real-time advanced Data Augmentation, and optimized hardware streaming.</b>
</p>

---
</div>

## 🚀 System Highlights

* **`MyConvBlock` Architecture**
  A custom modular engine fusing `Conv2d`, `BatchNorm2d`, `ReLU`, `Dropout`, and `MaxPool2d` into a single reusable layer for advanced feature extraction.

* **On-the-Fly Data Augmentation**
  Dynamic transformations (`Rotation`, `ResizedCrop`, `HorizontalFlip`, `ColorJitter`) executed inside the training loop to eliminate overfitting.

* **Normalized Data Management**
  A custom `Dataset` pipeline handling min-max scaling ($0-255 \to 0.0-1.0$) and spatial reshaping automatically.

* **Weight Serialization**
  An automated protocol that compiles and exports optimized structural weights directly into a deployable binary (`ultimate_asl_model.pth`).
---

## 🛠️ Complete Technical Stack

| Category | Technology / Library | Architectural Role |
| :--- | :--- | :--- |
| **Core Language** | `Python 3.x` | Base Environment & Scripting |
| **DL Framework** | `PyTorch (torch.nn)` | Neural Network Assembly, Tensors & Backpropagation |
| **Computer Vision**| `TorchVision` | Advanced Dynamic Transforms & Spatial Processing |
| **Data Engine** | `Pandas` | Fast Reading and Parsing of CSV Matrix Tables |
| **Hardware Driver**| `NVIDIA CUDA` | GPU Hardware Acceleration for Multi-Million Operations |


---
## 📊 Dataset Specifications (Sign Language MNIST)

The dataset maps rows of numbers into continuous spatial visuals. You can download the official dataset directly from the link below:

<div align="center">

### 📥 [Download Official ASL Dataset (ZIP)](https://storage.googleapis.com/download.tensorflow.org/data/sign_mnist.zip)

</div>

* **Training Examples:** 27,455 unique images.
* **Validation Examples:** 7,172 unique images.
* **Dimensionality:** Each image is a $28 \times 28$ pixel matrix.
* * **Classes:** 24 distinct classes corresponding to the alphabet (Letters **J** and **Z** are excluded as they require structural motion).

## 🧠 Model Architecture Blueprint

The neural engine sequentially narrows spatial resolutions to abstract high-level shapes from raw pixels before passing them into the fully connected classifier:

```text
       Input Mini-Batch (32 x 1 x 28 x 28)
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  MyConvBlock 1: Conv2d -> 25 Filters (3x3)   │  ──► Output: 32 x 25 x 14 x 14
└──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  MyConvBlock 2: Conv2d -> 50 Filters (3x3)   │  ──► Output: 32 x 50 x 7 x 7
│  * Implements 0.2 Dropout Protection         │
└──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  MyConvBlock 3: Conv2d -> 75 Filters (3x3)   │  ──► Output: 32 x 75 x 3 x 3
└──────────────────────────────────────────────┘
                       │
                       ▼
             [ nn.Flatten() Layer ]            ──► Transforms 3D Matrix to 1D Vector (675 features)
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Fully Connected Linear Layer (675 -> 512)   │  ──► Dense Structural Evaluation
│  * Implements 0.3 Dropout & ReLU Activation  │
└──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Final Output Linear Layer (512 -> 24)       │  ──► Yields 24 Logit Scores for Classification
└──────────────────────────────────────────────┘
