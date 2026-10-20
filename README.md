<div align="center">

# ⚡ SRGAN: Photo-Realistic Single Image Super-Resolution (4x)
### *Deep Learning Implementation in PyTorch & TensorLayerX*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorLayerX](https://img.shields.io/badge/TensorLayerX-0.5.8-00C49F?style=for-the-badge)](https://github.com/tensorlayer/TensorLayerX)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=for-the-badge)](LICENSE)
[![GitHub](https://img.shields.io/badge/Maintained%20by-Aditya%20Arote-black?style=for-the-badge&logo=github)](https://github.com/AdityaArote)

<p align="center">
  A state-of-the-art Generative Adversarial Network that synthesizes high-frequency, photorealistic 4x textures from low-resolution images.
</p>

<img src="img/model.jpeg" width="90%" alt="SRGAN Architecture" />

</div>

---

## 🌟 Overview

Single Image Super-Resolution (SISR) has traditionally relied on minimizing Mean Squared Error (MSE), which maximizes Peak Signal-to-Noise Ratio (PSNR) but produces unnaturally smooth, blurry images lacking fine details.

This project implements **SRGAN** (*Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network* by Christian Ledig et al.). By coupling deep **Residual Blocks**, **Sub-pixel Convolutions (PixelShuffle)**, **VGG-19 Perceptual Content Loss**, and an **Adversarial Loss**, the network recovers fine photographic details and microscopic textures.

### 🔬 Results Showcase
<div align="center">
  <img src="img/SRGAN_Result3.png" width="90%" alt="SRGAN Super-Resolution Visual Comparison" />
</div>

---

## 🚀 Key Features

- **4x Super-Resolution**: Upscales low-resolution images by a factor of 4 while generating sharp, believable textures.
- **Deep ResNet Architecture**: Generator with 16 Residual Blocks using identical identity skip connections.
- **PixelShuffle Upsampling**: Sub-pixel convolutions avoid checkerboard artifacts common to transposed deconvolution.
- **VGG-19 Perceptual Loss**: Measures distance in high-level feature space rather than raw pixel Euclidean distance.
- **Multi-Backend Ready**: Optimized for **PyTorch** (compatible with Apple Silicon MPS, NVIDIA CUDA, and CPU) alongside support for TensorFlow.
- **Instant CLI Demo**: Ready-to-use inference script (`demo.py`) for processing arbitrary images with single commands.

---

## 🧠 Model Architecture & Mechanism

```
Low-Res Input (96x96x3)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  GENERATOR (SRGAN_g)                                   │
│  ├── Initial Conv2D (64 filters, 3x3, PReLU/ReLU)      │
│  ├── 16 Residual Blocks with Identity Skip Connections │
│  ├── 2x PixelShuffle Upsample Blocks (2x * 2x = 4x)    │
│  └── Final Conv2D + Tanh [-1.0, 1.0]                   │
└────────────────────────────────────────────────────────┘
       │
       ▼ Generated Super-Resolution (384x384x3)
       ├─────────────────────────────────┐
       ▼                                 ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│ DISCRIMINATOR (SRGAN_d)   │     │ VGG-19 Feature Extractor  │
│ ├── Strided Conv Blocks   │     │ ├── Pretrained on ImageNet│
│ ├── LeakyReLU Activations │     │ ├── Evaluates deep shapes │
│ └── Sigmoid Output [0, 1] │     │ └── Computes Content Loss │
└───────────────────────────┘     └───────────────────────────┘
```

### Why These Activations?
- **LeakyReLU**: Employed across the Discriminator to prevent the *"Dying ReLU"* trap where inactive neurons get zero gradient.
- **Sigmoid**: Compresses discriminator output into $[0.0, 1.0]$ representing probability of realism.
- **Tanh**: Normalizes final generator output into $[-1.0, 1.0]$, ensuring valid bounded pixel color ranges.

### Multi-Task Loss Formulation
The Generator is trained using a weighted composite loss:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{MSE}} + 2\cdot 10^{-6} \mathcal{L}_{\text{VGG}} + 10^{-3} \mathcal{L}_{\text{Adversarial}}$$

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AdityaArote/SRGAN-SuperResolution.git
cd SRGAN-SuperResolution
```

### 2. Create and Activate Conda Environment
```bash
conda create -y -n srgan python=3.10
conda activate srgan
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Download Pre-trained Weights

Pretrained generator weights can be downloaded into the `models/` directory:

```bash
mkdir -p models
# Download pre-trained generator weights via gdown
gdown 1GlU9At-5XEDilgnt326fyClvZB_fsaFZ -O models/g.npz
```

*(Optional) VGG-19 weights for training can be downloaded from [Google Drive](https://drive.google.com/file/d/1CLw6Cn3yNI1N15HyX99_Zy9QnDcgP3q7/view?usp=sharing) and placed at `model/vgg19.npy`.*

---

## 💻 Usage

### 1. Launch Interactive Web Application (ApexSR UI)
Experience real-time interactive super-resolution with dual-pane comparison, split-slider, preset samples, and metrics:

```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 2. Run 4x Super-Resolution CLI (Demo)
Run inference on a custom image using [demo.py](demo.py):

```bash
python demo.py --image "path/to/your/image.jpg" --output_dir samples/
```

This generates:
- **`[name]_lr.png`**: Original input.
- **`[name]_bicubic.png`**: Standard 4x bicubic baseline.
- **`[name]_srgan.png`**: Photorealistic 4x SRGAN result.

### 3. Evaluation Mode
Run evaluation against validation images or fallback samples:

```bash
python train.py --mode=eval
```

### 4. Training on DIV2K or Custom Dataset
1. Configure dataset paths in [config.py](config.py):
   ```python
   config.TRAIN.hr_img_path = 'DIV2K/DIV2K_train_HR/'
   ```
2. Start the two-phase training loop (warm-up MSE initialization followed by adversarial training):
   ```bash
   python train.py --mode=train
   ```

---

## 📁 Repository Structure

```
SRGAN-SuperResolution/
├── app.py              # Flask server backend for web application
├── static/             # Web UI assets (HTML, CSS, JS, sample presets)
├── config.py           # Training and validation hyperparameters
├── demo.py             # CLI inference script for custom images
├── srgan.py            # SRGAN Generator (SRGAN_g) and Discriminator (SRGAN_d)
├── train.py            # Training and evaluation loops
├── vgg.py              # Pre-trained VGG-19 perceptual network
├── requirements.txt    # Project dependencies
├── img/                # Architecture diagrams and benchmark samples
├── models/             # Checkpoints and pre-trained weights (gitignored)
└── samples/            # Output high-resolution images (gitignored)
```

---

## 📚 References & Citation

- **Original Paper**: Ledig, C., Theis, L., Huszár, F., et al. (2017). *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*. CVPR 2017. [arXiv:1609.04802](https://arxiv.org/abs/1609.04802).
- **Framework**: Built with [TensorLayerX](https://github.com/tensorlayer/TensorLayerX) & [PyTorch](https://pytorch.org/).

---

## 👤 Author
**Aditya Arote**  
GitHub: [@AdityaArote](https://github.com/AdityaArote)
