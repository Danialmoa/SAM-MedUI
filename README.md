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
## Citation

If you use SAM-MedUI in your research, please cite:

```bibtex
@article{moafi2024sammedui,
  title={Interactive Deep Learning for Myocardial Scar Segmentation Using Cardiovascular Magnetic Resonance},
  author={Moafi, Aida and Moafi, Danial and Shergil, Simran and Mirkes, Evgeny M. and Adlam, David and Samani, Nilesh J. and McCann, Gerry P. and Ghazi, Mostafa Mehdipour and Arnold, J. Ranjit},
  journal={},
  year={}
}
```

**Authors:**  
Aida Moafi¹, Danial Moafi², Simran Shergil¹, Evgeny M. Mirkes³, David Adlam¹⁵, Nilesh J. Samani¹⁵, Gerry P. McCann¹⁵, Mostafa Mehdipour Ghazi⁴*, J. Ranjit Arnold¹*

*Joint senior authorship

**Affiliations:**  
¹ Department of Cardiovascular Sciences, University of Leicester, NIHR Leicester Biomedical Research Centre and BHF Centre of Research Excellence, Glenfield Hospital, Leicester, UK  
² Department of Information Engineering and Mathematics, University of Siena, Siena, Italy  
³ Department of Mathematics, University of Leicester, Leicester, UK  
⁴ Pioneer Centre for AI, Department of Computer Science, University of Copenhagen, Copenhagen, Denmark  
⁵ Centre for Digital Health and Precision Medicine, University of Leicester

---

## Acknowledgments

We thank the following projects and teams for their foundational work:

- [Meta AI Research](https://github.com/facebookresearch/segment-anything) for developing and open-sourcing the **Segment Anything Model (SAM)**
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the **YOLOv12** object detection framework

---

## Contact

For questions and collaborations, please contact:

- **Aida Moafi** — [am1392@leicester.ac.uk](mailto:am1392@leicester.ac.uk)
- **Danial Moafi** — [d.moafi@student.unisi.it](mailto:d.moafi@student.unisi.it)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 




