# 🤖 Sign Language Digit Recognition using Deep CNN

<div align="center">

![License](https://img.shields.io/badge/license-Academic%20Reference-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Accuracy](https://img.shields.io/badge/accuracy-97.58%25-success.svg)

*A comprehensive deep learning project implementing CNN for sign language digit recognition with extensive hyperparameter optimization and academic-grade documentation.*

</div>

## 📋 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [📁 Project Structure](#-project-structure)
- [📊 Dataset Information](#-dataset-information)
- [🏗️ Model Architecture](#️-model-architecture)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Hyperparameter Experiments](#️-hyperparameter-experiments)
- [📈 Results & Performance](#-results--performance)
- [📚 Academic Documentation](#-academic-documentation)
- [🔧 Installation & Requirements](#-installation--requirements)
- [📝 Usage Instructions](#-usage-instructions)
- [🎨 Visualizations](#-visualizations)
- [⚠️ Important Notes](#️-important-notes)

## 🎯 Project Overview

This project develops a **state-of-the-art Convolutional Neural Network** for recognizing hand gestures corresponding to digits 0-9 in sign language. The implementation features:

- ✅ **Custom CNN architecture** with progressive filter scaling
- ✅ **Comprehensive hyperparameter optimization** (11 different configurations)
- ✅ **IEEE conference paper format** technical documentation
- ✅ **Extensive evaluation metrics** (Accuracy, Precision, Recall, F1, AUC-ROC)
- ✅ **Professional visualizations** and analysis plots
- ✅ **CPU-optimized training** for accessibility

## 📁 Project Structure

```
SignLanguage/
├── 📂 data/
│   ├── X.npy                          # Input images (2,062 × 64×64 grayscale)
│   └── Y.npy                          # One-hot encoded labels (2,062 × 10)
│
├── 📂 images/                         # Generated visualizations
│   ├── comprehensive_analysis.png     # Complete hyperparameter analysis
│   ├── confusion_matrix_baseline.png  # Baseline model confusion matrix
│   ├── dataset_samples.png           # Dataset sample images
│   ├── sample_images.png             # Class distribution samples
│   └── training_histories.png        # Training/validation curves
│
├── 🐍 cpu_cnn_experiments.py         # Main implementation & experiments
├── 🔍 investigate_data.py            # Dataset exploration script
├── 📄 README.md                      # This comprehensive guide
├── ⚖️ LICENSE                        # Academic Reference License
└── 🚫 .gitignore                     # Git ignore configuration
```

> **Note**: LaTeX files, virtual environments, and PDFs are excluded from the repository as per `.gitignore` configuration.

## 📊 Dataset Information

<div align="center">

| **Attribute** | **Value** |
|---------------|-----------|
| 📸 **Total Samples** | 2,062 images |
| 🖼️ **Image Dimensions** | 64×64 pixels (grayscale) |
| 🏷️ **Classes** | 10 (digits 0-9) |
| ⚖️ **Distribution** | Well-balanced (~206 samples per class) |
| 💾 **Format** | NumPy arrays (.npy) |
| 📏 **Input Shape** | (2062, 64, 64, 1) |
| 🎯 **Output Shape** | (2062, 10) - One-hot encoded |

</div>

### Dataset Characteristics:
- 🔍 **Preprocessing**: Normalized pixel values (0-1 range)
- 🎲 **Data Split**: 80% training, 20% validation (stratified)
- 🔄 **Augmentation**: None (clean, consistent dataset)
- ✅ **Quality**: High-quality grayscale images with clear gestures

## 🏗️ Model Architecture

### 🧠 CNN Architecture Design:

```
Input (64×64×1)
      ↓
┌─────────────────┐
│ Conv Block 1    │ ← 32 filters, 3×3, ReLU + BatchNorm + Dropout + MaxPool
├─────────────────┤
│ Conv Block 2    │ ← 64 filters, 3×3, ReLU + BatchNorm + Dropout + MaxPool  
├─────────────────┤
│ Conv Block 3    │ ← 128 filters, 3×3, ReLU + BatchNorm + Dropout + MaxPool
├─────────────────┤
│ Flatten         │
├─────────────────┤
│ Dense (256)     │ ← ReLU + Dropout
├─────────────────┤
│ Dense (128)     │ ← ReLU + Dropout
├─────────────────┤
│ Output (10)     │ ← Softmax
└─────────────────┘
```

### 🔧 Key Components:
- **🎯 Activation Functions**: ReLU (hidden layers), Softmax (output)
- **🛡️ Regularization**: Dropout (0.2-0.5), L1/L2 weight decay, Batch Normalization
- **⏰ Early Stopping**: Patience-based training termination
- **📉 Optimizer**: Adam with adaptive learning rate
- **📊 Loss Function**: Categorical Crossentropy

## 🚀 Quick Start

### 1️⃣ **Clone and Setup**
```bash
git clone https://github.com/Muhammad-Hamdan-Rauf/sign-language-digit-cnn
cd sign-language-digit-cnn
```

### 2️⃣ **Install Dependencies**
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
```

### 3️⃣ **Run Experiments**
```bash
python cnn_experiments.py
```

### 4️⃣ **Explore Data** (Optional)
```bash
python investigate_data.py
```

## ⚙️ Hyperparameter Experiments

<div align="center">

### 🧪 **11 Comprehensive Experiments Conducted**

| **Parameter** | **Values Tested** | **Best Value** |
|---------------|-------------------|----------------|
| 🔢 **Batch Size** | [16, 32, 64] | **16-32** |
| 📈 **Learning Rate** | [0.0005, 0.001, 0.002] | **0.001** |
| 💧 **Dropout Rate** | [0.2, 0.3, 0.5] | **0.3** |
| 🔒 **L1 Regularization** | [0.0, 0.001] | **0.001** |
| 🔒 **L2 Regularization** | [0.001, 0.005] | **0.005** |
| ⏱️ **Early Stopping** | [5, 15, 20] | **15-20 epochs** |

</div>

### 📋 **Experiment Configurations:**
1. **Baseline Model** - Default parameters
2. **Batch Size Variations** - 16, 32, 64
3. **Learning Rate Tuning** - 0.0005, 0.001, 0.002  
4. **Dropout Optimization** - 0.2, 0.3, 0.5
5. **L1 Regularization** - λ = 0.001
6. **L2 Regularization** - λ = 0.005
7. **Early Stopping** - Patience variations

## 📈 Results & Performance

### 🏆 **Top Performing Models:**

<div align="center">

| **Rank** | **Model** | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC-ROC** |
|----------|-----------|--------------|---------------|------------|--------------|-------------|
| 🥇 | **Baseline** | **97.58%** | **97.61%** | **97.58%** | **97.58%** | **99.98%** |
| 🥈 | **Batch Size 16** | **96.85%** | **96.85%** | **96.85%** | **96.85%** | **99.93%** |
| 🥉 | **L2 Regularization** | **96.61%** | **96.78%** | **96.59%** | **96.59%** | **99.94%** |
| 4️⃣ | **L1 Regularization** | **96.13%** | **96.17%** | **96.12%** | **96.10%** | **99.91%** |

</div>

### 💡 **Key Insights:**
- ✅ **Exceptional Performance**: 97.58% accuracy achieved
- ✅ **Robust Generalization**: High AUC-ROC scores (>99.9%)
- ✅ **Balanced Metrics**: Consistent precision, recall, and F1-scores
- ✅ **Optimal Configuration**: Moderate batch sizes and dropout rates work best

## 📚 Academic Documentation

This project includes comprehensive academic documentation following **IEEE conference paper standards**:

### 📄 **Report Sections:**
1. **📝 Abstract** - Methodology and results summary
2. **🎯 Introduction** - Problem statement and motivation  
3. **📖 Related Work** - Literature review and background
4. **🔬 Methodology** - Dataset, preprocessing, and architecture
5. **📊 Experimental Results** - Comprehensive analysis and comparisons
6. **💭 Discussion** - Insights and practical implications
7. **⚠️ Limitations** - Current constraints and future work
8. **🎯 Conclusion** - Key findings and contributions

> **Note**: LaTeX source files are kept private but the methodology and results are fully documented in this README.

## 🔧 Installation & Requirements

### 📋 **System Requirements:**
- 🐍 **Python**: 3.8 or higher
- 💾 **RAM**: 8GB+ recommended  
- ⚡ **CPU**: Multi-core processor (GPU not required but recommended)
- 💿 **Storage**: 500MB+ free space

### 📦 **Dependencies:**
```bash
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

### 🚀 **Installation Command:**
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
```

## 📝 Usage Instructions

### 🎯 **Primary Script: `cnn_experiments.py`**

This script performs the complete experimental pipeline:

1. **📊 Data Loading & Exploration**
2. **🏗️ Model Architecture Setup**
3. **🎯 Baseline Model Training**
4. **⚙️ Hyperparameter Experiments** (with user prompt)
5. **📈 Comprehensive Evaluation**
6. **🎨 Visualization Generation**

### 🔍 **Secondary Script: `investigate_data.py`**

Dedicated dataset exploration and analysis:
- 📊 Dataset statistics and distribution
- 🖼️ Sample image visualization
- 📈 Class balance analysis
- 💾 Data quality assessment

### ⌨️ **Execution Examples:**

```bash
# Run complete experimental suite
python cnn_experiments.py

# Explore dataset characteristics
python investigate_data.py

# View generated visualizations
# Check the images/ directory for PNG files
```

## 🎨 Visualizations

The project generates comprehensive visualizations stored in the `images/` directory:

### 📊 **Available Plots:**

| **Visualization** | **Description** | **Key Insights** |
|-------------------|-----------------|------------------|
| 🖼️ `sample_images.png` | Dataset samples by class | Visual diversity and quality |
| 📈 `training_histories.png` | Training/validation curves | Convergence and overfitting analysis |
| 🎯 `comprehensive_analysis.png` | Complete hyperparameter analysis | Performance comparisons |
| 🔍 `confusion_matrix_baseline.png` | Classification breakdown | Per-class performance details |
| 📊 `dataset_samples.png` | Class distribution samples | Balance and representation |

### 🎨 **Visualization Features:**
- 📊 Professional matplotlib/seaborn styling
- 🎨 Clear legends and annotations
- 📐 High-resolution output (300 DPI)
- 🎯 Comprehensive metric coverage
- 📈 Interactive-style layouts

## ⚠️ Important Notes

### 📜 **License and Usage:**
- 📖 **Academic Reference License**: View-only access for educational purposes
- 🚫 **No Redistribution**: Code cannot be copied or reused
- 🎓 **Academic Integrity**: Using this code in assignments may constitute plagiarism
- ✅ **Citation Required**: Appropriate attribution needed for academic references

### 🔐 **Repository Configuration:**
- 🚫 Virtual environments (`venv/`) excluded from repository
- 🚫 LaTeX files (`.tex`, `.cls`, `.sty`) kept private  
- 🚫 Generated PDFs excluded from version control
- ✅ Core implementation and visualizations included

### 🎯 **Educational Purpose:**
This project demonstrates best practices in:
- 🧠 **Deep Learning**: CNN architecture design and optimization
- 📊 **Experimental Design**: Systematic hyperparameter exploration
- 📝 **Academic Writing**: IEEE format technical documentation
- 💻 **Software Engineering**: Clean code organization and documentation
- 📈 **Data Analysis**: Comprehensive evaluation and visualization

---

<div align="center">

### 🎓 **Academic Excellence | 🤖 Deep Learning | 📊 Data Science**

*Developed as part of Generative AI coursework - Semester 7*

**⭐ If this repository helped your understanding, please star it! ⭐**

</div>