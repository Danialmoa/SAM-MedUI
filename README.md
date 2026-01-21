# SAM-MedUI: IInteractive Deep Learning for Myocardial Scar Segmentation Using Cardiovascular Magnetic Resonance

SAM-MedUI is an interactive tool for myocardial scar segmentation from late gadolinium enhancement cardiac MRI (LGE-CMR). It combines a fine-tuned Segment Anything Model with YOLO-based detection and a clinician-facing GUI, enabling fast and reproducible scar quantification.
## Screenshot
<img width="789" height="895" alt="image" src="https://github.com/user-attachments/assets/a259853c-8ea1-427a-8eea-fdd465bb1559" />


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

## Quick Start
### Installation

# Clone the repository
git clone https://github.com/Danialmoa/SAM-MedUI
cd SAM-MedUI

# Install dependencies
pip install -r requirements.txt

# Create checkpoints folder
mkdir checkpoints

### Download Model Weights

Place the following files in the `checkpoints/` folder:

| File | Description |
|------|-------------|
| `sam_vit_b_01ec64.pth` | SAM base model |
| `best_model.pth` | Fine-tuned SAM model |
| `yolo_best.pt` | YOLO detection model |

# Run the GUI
cd GUI
python main.py


### Requirements

- Python 3.8+
- PyTorch 2.7.0
- Training (Fine-tuning): CUDA-capable GPU required
- GUI Application: CPU or GPU supported - can run on any laptop
- **Note**: The GUI application can run on CPU for inference, making it accessible on standard laptops. GPU is only required for training/fine-tuning the models.

## Usage
### Demo
![JCMR_Video-ezgif com-optimize](https://github.com/user-attachments/assets/8b4b2ff4-a984-4f0e-8444-ed149fb882a6)

### GUI Workflow

1. Load images → **Load Folder** or **Load Files**
2. Add prompts → Click for points, drag for bounding box, or use **Auto-Detect**
3. Generate → Click **Generate Segmentation**
4. Refine → Use **Expand** or **Shrink** if needed
5. Save → **Save Mask** or **Export Results**

## Project Structure

```
SAM-MedUI/
├── GUI/
│   ├── main.py
│   ├── model_handler.py
│   ├── canvas_view.py
│   └── thumbnail_gallery.py
├── SAM_finetune/
│   ├── models/
│   ├── train/
│   └── utils/
├── checkpoints/
├── requirements.txt
└── README.md
```

 



