# SAM-MedUI: IInteractive Deep Learning for Myocardial Scar Segmentation Using Cardiovascular Magnetic Resonance

SAM-MedUI is an interactive tool for myocardial scar segmentation from late gadolinium enhancement cardiac MRI (LGE-CMR). It combines a fine-tuned Segment Anything Model with YOLO-based detection and a clinician-facing GUI, enabling fast and reproducible scar quantification.

## Features

- **Interactive GUI**: User-friendly interface built with Tkinter for real-time medical image segmentation
- **Multiple Input Formats**: Supports DICOM, NIfTI, and standard image formats (JPEG, PNG)
- **Flexible Prompting**: 
  - Point-based prompts (foreground points to indicate regions of interest)
  - Bounding box prompts
  - Automatic detection using YOLO integration
- **Batch Processing**: Process multiple images with thumbnail gallery navigation
- **Fine-tuning Capabilities**: Tools for fine-tuning SAM on medical imaging datasets
- **Advanced Features**:
  - Real-time mask visualization
  - Morphological operations (expand/shrink masks)
  - Confidence threshold adjustment
  - Gamma correction for image enhancement
  - Export segmentation results to CSV
  - Undo/redo functionality
  - Zoom and pan for detailed inspection

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.7.0
- Training (Fine-tuning): CUDA-capable GPU required
- GUI Application: CPU or GPU supported - can run on any laptop
- **Note**: The GUI application can run on CPU for inference, making it accessible on standard laptops. GPU is only required for training/fine-tuning the models.

### Setup

1. Clone the repository:
git clone <repository-url>
cd SAM-MedUI

