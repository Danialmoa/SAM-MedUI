<div align="center">

# SAM-MedUI

### Interactive Deep Learning for Myocardial Scar Segmentation Using Cardiovascular Magnetic Resonance

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/JCMR-Paper-blue)](https://www.sciencedirect.com/science/article/pii/S1097664726000384?via%3Dihub)

**SAM-MedUI** is a clinician-friendly interactive tool for myocardial scar segmentation from Late Gadolinium Enhancement Cardiac MRI (LGE-CMR). It combines a fine-tuned [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) with YOLO-based automatic detection and an intuitive GUI—enabling fast, accurate, and reproducible scar quantification without requiring any coding knowledge.

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Usage](#usage) • [Architecture](#architecture) • [Citation](#citation)

</div>

---

## Screenshot

<p align="center">
  <img width="789" alt="SAM-MedUI Interface" src="https://github.com/user-attachments/assets/a259853c-8ea1-427a-8eea-fdd465bb1559" />
</p>

---

## Key Highlights

| Feature | Description |
|---------|-------------|
| **Clinical-Grade Interface** | Built specifically for clinicians with intuitive controls and real-time feedback |
| **Multiple Input Formats** | Native support for DICOM, NIfTI (3D volumes), JPEG, PNG, and BMP |
| **Flexible Prompting** | Point-based, bounding box, and automatic YOLO detection |
| **Real-time Refinement** | Morphological operations, confidence adjustment, and undo |
| **Quantitative Analysis** | Automatic pixel mass calculations using DICOM/NIfTI metadata |
| **Runs on CPU** | GUI inference works on any laptop—no GPU required |

---

## Features

### Interactive Segmentation
- **Point Prompts**: Left-click to add foreground points (green)
- **Bounding Box Prompts**: Click and drag to define regions of interest with adjustable handles
- **Auto-Detection**: YOLO-based automatic cardiac region detection reduces manual prompting

### Medical Imaging Support
- **DICOM**: Full metadata extraction (patient ID, pixel spacing, slice thickness)
- **NIfTI**: 3D volume support with automatic slice extraction and navigation
- **Standard Formats**: JPEG, PNG, BMP for preprocessed images

### Real-time Visualization
- **Mask Overlay**: Alpha-blended segmentation visualization
- **Gamma Correction**: Adjustable contrast (0.2–1.7) for enhanced visibility
- **Zoom & Pan**: 0.5x to 5.0x magnification with smooth navigation

### Mask Refinement
- **Morphological Operations**: Expand/shrink masks with configurable iterations
- **Confidence Threshold**: Dynamic adjustment (0.3–0.99) with real-time preview
- **Undo**: Up to 10 levels of operation history

### Batch Processing & Export
- **Thumbnail Gallery**: Patient-centric navigation with multi-slice support
- **Batch Save**: Export all masks with a single click
- **CSV Export**: Quantitative results including patient ID, slice, and scar mass
- **Prompt Storage**: JSON-based prompt saving for reproducibility

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Danialmoa/SAM-MedUI.git
cd SAM-MedUI
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run

Model weights are **downloaded automatically** on first launch from [🤗 Hugging Face](https://huggingface.co/AidaAIDL/SAM_MEDUI).

```bash
cd GUI
python main.py
```

To download weights manually instead:

```bash
pip install huggingface_hub
huggingface-cli download AidaAIDL/SAM_MEDUI --local-dir checkpoints/
```

| File | Description |
|------|-------------|
| `best_model.pth` | Fine-tuned SAM for cardiac scar segmentation |
| `yolo_best.pt` | YOLO detection model for automatic ROI detection |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **GPU (Training)** | CUDA-capable GPU | NVIDIA GPU with 8+ GB VRAM |
| **GPU (Inference)** | Not required | Optional for faster inference |
| **Storage** | 2 GB for models | 5+ GB with datasets |

---

## Quick Start

```bash
cd GUI
python main.py
```

### Demo

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b4b2ff4-a984-4f0e-8444-ed149fb882a6" alt="SAM-MedUI Demo" />
</p>

---

## Usage

### Basic Workflow

1. **Load Images** → Click `Load Folder` or `Load Files` to import medical images
2. **Add Prompts** → Click for points, drag for bounding boxes, or use `Auto-Detect`
3. **Generate** → Click `Generate Segmentation` to create the mask
4. **Refine** → Adjust threshold or use `Expand`/`Shrink` for fine-tuning
5. **Save** → Export with `Save Mask`, `Save All Masks`, or `Export Results`

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `←` / `→` | Navigate between images |
| `Ctrl+Z` | Undo last operation |
| `Ctrl++` / `Ctrl+-` | Zoom in / out |
| `Ctrl+0` | Reset zoom |
| `Ctrl+Arrow keys` | Pan view |
| Hold `Z` | Temporarily hide mask & prompts |

### Prompting Modes

| Mode | How to Use | Best For |
|------|------------|----------|
| **Point (Foreground)** | Left-click on target region | Precise selection of scar tissue |
| **Bounding Box** | Click and drag rectangle | Defining region of interest |
| **Auto-Detect** | Click the Auto-Detect button | Quick initial detection |

### Export Options

- **Save Mask**: Export current segmentation as PNG
- **Save All Masks**: Batch export all processed images
- **Export Results**: Generate CSV with quantitative metrics:
  - Patient ID
  - Image/slice name
  - Scar mass (calculated from pixel spacing and slice thickness)

---

## Architecture

### Project Structure

```
SAM-MedUI/
├── GUI/                              # Main Application
│   ├── main.py                       # GUI entry point and main window
│   ├── model_handler.py              # SAM & YOLO inference logic
│   ├── canvas_view.py                # Image display and annotation
│   ├── thumbnail_gallery.py          # Patient navigation and thumbnails
│   └── download_weights.py           # Auto-download weights from HuggingFace
│
├── SAM_finetune/                     # Training Pipeline
│   ├── models/
│   │   ├── sam_model.py              # SAM wrapper with fine-tuning support
│   │   ├── dataset.py                # Medical imaging dataset loader
│   │   ├── loss.py                   # Combined loss function
│   │   └── prompt_generator.py       # Bounding box & point generation
│   │
│   ├── train/
│   │   └── trainer.py                # Training loop with W&B logging
│   │
│   └── utils/
│       ├── config.py                 # Configuration dataclasses
│       ├── logger_func.py            # Logging setup with rotation
│       ├── preprocessing.py          # Image preprocessing utilities
│       ├── z_score_norm.py           # Percentile normalization
│       └── visualize.py              # Visualization helpers
│
├── checkpoints/                      # Model weights (auto-downloaded)
├── logs/                             # Application logs
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
└── README.md
```

### Segmentation Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ Load Image  │───▶│ Add Prompts  │───▶│ SAM Forward │───▶│ Apply Mask   │
│ (DICOM/NIfTI)    │ (Points/BBox) │    │   Pass      │    │  Threshold   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          │                                       │
                   ┌──────▼──────┐                         ┌──────▼──────┐
                   │ YOLO Auto-  │                         │ Morphological│
                   │  Detection  │                         │  Refinement  │
                   └─────────────┘                         └─────────────┘
```

### Training Pipeline

The fine-tuning pipeline includes:

- **Medical-specific augmentations** via TorchIO (elastic deformation, motion artifacts, bias field)
- **Combined loss function**: Dice + BCE + Soft BCE + KL Divergence + Diversity Loss
- **Experiment tracking** with Weights & Biases
- **Learning rate scheduling** with Cosine Annealing

---

## Training (Fine-tuning)

To fine-tune SAM on your own cardiac MRI dataset:

```python
from SAM_finetune.utils.config import SAMFinetuneConfig, SAMDatasetConfig
from SAM_finetune.models.dataset import SAMDataset
from SAM_finetune.train.trainer import SAMTrainer

# Configure dataset
dataset_config = SAMDatasetConfig(
    dataset_path="path/to/dataset",
    point_prompt=True,
    box_prompt=True,
    number_of_prompts=2,
)

# Create dataset
train_dataset = SAMDataset(config=dataset_config)

# Configure training
train_config = SAMFinetuneConfig(
    sam_path="pretrained_models/sam_vit_b_01ec64.pth",
    learning_rate=1e-4,
    num_epochs=100,
    batch_size=4,
)

# Start training
trainer = SAMTrainer(config=train_config, train_dataset=train_dataset)
trainer.train()
```

---

## Citation

If you use SAM-MedUI in your research, please cite:

```bibtex
@article{moafi2026sammedui,
  title={Interactive Deep Learning for Myocardial Scar Segmentation Using Cardiovascular Magnetic Resonance},
  author={Moafi, Aida and Moafi, Danial and Shergil, Simran and Mirkes, Evgeny M. and Adlam, David and Samani, Nilesh J. and McCann, Gerry P. and Ghazi, Mostafa Mehdipour and Arnold, J. Ranjit},
  journal={Journal of Cardiovascular Magnetic Resonance},
  year={2026},
  publisher={Elsevier},
  url={https://www.sciencedirect.com/science/article/pii/S1097664726000384}
}

```

### Authors

**Aida Moafi**¹, **Danial Moafi**², **Simran Shergil**¹, **Evgeny M. Mirkes**³, **David Adlam**¹⁵, **Nilesh J. Samani**¹⁵, **Gerry P. McCann**¹⁵, **Mostafa Mehdipour Ghazi**⁴\*, **J. Ranjit Arnold**¹\*

*\* Joint senior authorship*

### Affiliations

¹ Department of Cardiovascular Sciences, University of Leicester, NIHR Leicester Biomedical Research Centre and BHF Centre of Research Excellence, Glenfield Hospital, Leicester, UK
² Department of Information Engineering and Mathematics, University of Siena, Siena, Italy
³ Department of Mathematics, University of Leicester, Leicester, UK
⁴ Pioneer Centre for AI, Department of Computer Science, University of Copenhagen, Copenhagen, Denmark
⁵ Centre for Digital Health and Precision Medicine, University of Leicester

---

## Acknowledgments

We gratefully acknowledge the following projects:

- [**Segment Anything (SAM)**](https://github.com/facebookresearch/segment-anything) by Meta AI Research
- [**Ultralytics YOLO**](https://github.com/ultralytics/ultralytics) for object detection
- [**TorchIO**](https://github.com/fepegar/torchio) for medical image augmentation

---

## Contact

For questions or collaborations:


**Aida Moafi**  [am1392@leicester.ac.uk](mailto:am1392@leicester.ac.uk) 
**Danial Moafi**  [d.moafi@student.unisi.it](mailto:d.moafi@student.unisi.it)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with care for the medical imaging community**

</div>

