# Deepfake Generation & Detection

A deep learning project that explores both **generating** and **detecting** deepfake face images. The project is split into two Jupyter Notebooks, one for training generative models and another for training and evaluating binary classifiers.

## Project Structure

```
.
├── deep_fake_generators.ipynb   # Notebook for deepfake image generation
├── deep_fake_classifiers.ipynb  # Notebook for deepfake detection / classification
├── data/                        # Dataset directory (see Data Setup below)
│   ├── wiki/                    # Real face images (100 subfolders: 00-99)
│   ├── inpainting/              # Fake images - inpainting method
│   ├── insight/                 # Fake images - face-swap (InsightFace)
│   └── text2img/                # Fake images - text-to-image generation
├── models/                      # Models code (.py)
│   ├── classifiers/             # Classifier (CNN/, NN/, ViT/)
│   └── generators/              # Generator (Diffusion/, GAN/)
├── experiment_logs/             # Training logs, metrics, and plots
│   ├── classifiers/             # Logs split by augmentation/no_augmentation
│   │   ├── no_augmentation/     # One subfolder per model variant
│   │   └── augmentation/        # One subfolder per model variant
│   └── generators/              # One subfolder per generator version
│       ├── DCGAN_v1/
│       ├── DCGAN_v2/
│       ├── DCGAN_v3/
│       └── DDPM_v4/
└── utils/
    └── gradcam.py               # Grad-CAM visualization utilities
```

## Notebooks

### 1. `deep_fake_generators.ipynb` - Deepfake Generation

Trains generative models to synthesize realistic face images. Four iterations are explored:

| Version | Model | Key Idea |
|---------|-------|----------|
| DCGAN v1 | DCGAN with BCE loss | Baseline GAN |
| DCGAN v2 | DCGAN with constrained discriminator | Addressing discriminator dominance |
| DCGAN v3 | WGAN-GP | Wasserstein loss + gradient penalty for stable training |
| DDPM v4 | Denoising Diffusion Probabilistic Model | U-Net denoiser with MTCNN face cropping |

### 2. `deep_fake_classifiers.ipynb` - Deepfake Detection

Trains and evaluates binary classifiers (real vs. fake) across multiple architectures, dataset splits, and augmentation strategies.

**Architectures:**

| Family | Model | Notes |
|--------|-------|-------|
| NN | NN_1 | Fully-connected baseline on flattened 64×64 inputs |
| CNN | ResNet34 | Residual network, frozen & full fine-tuning |
| CNN | EfficientNet-B0 | Compound-scaled CNN, frozen & full fine-tuning |
| CNN | Xception | Depthwise separable convolutions, frozen & full fine-tuning |
| ViT | ViT-B/16 | Vision Transformer, frozen & full fine-tuning |

Each architecture is trained on four dataset splits: **Inpainting**, **Insight**, **Text2Img**, and **All Fakes**, both **with and without data augmentation**. The notebook also includes Grad-CAM visualizations and cross-model comparison charts.

## Data Setup

Place your dataset inside the `data/` directory following the structure below. Each subfolder should contain face images organized into numbered subfolders (`00`-`99`):

```
data/
├── wiki/          # Real face images
├── inpainting/    # Fake - inpainting-based
├── insight/       # Fake - face-swap (InsightFace)
└── text2img/      # Fake - text-to-image
```

> **Note:** The dataset is not included in this repository. You will need to obtain and place the images manually.

## How to Run

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support recommended)
- Jupyter Notebook or JupyterLab
- Additional dependencies: `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `facenet-pytorch` (for MTCNN in the generators notebook)

### Running the Notebooks

1. **Clone the repository** and place the dataset in `data/` (see above).

2. **Install dependencies:**
   ```bash
   pip install torch torchvision scikit-learn matplotlib seaborn facenet-pytorch
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Open and run the desired notebook:**
   - `deep_fake_generators.ipynb` - to train generative models.
   - `deep_fake_classifiers.ipynb` - to train and evaluate classifiers.

5. Run cells **sequentially from top to bottom**. Each notebook is self-contained.

> **Tip:** A GPU is highly recommended. The notebooks automatically detect and use CUDA, MPS (Apple Silicon), or CPU.

## Where Results Are Saved

- **Training logs , plots and model weights** → `experiment_logs/classifiers/` and `experiment_logs/generators/`
  - Classifier logs are further split by `augmentation/` and `no_augmentation/`, with a subfolder per model variant (e.g., `ResNet34_full/`, `ViT_B_16_frozen/`).
  - Generator logs have a subfolder per iteration (`DCGAN_v1/`, `DCGAN_v2/`, `DCGAN_v3/`, `DDPM_v4/`).
