import tkinter as tk
from ttkbootstrap import Canvas, Scrollbar, Frame, Label, Style, Button
from PIL import Image, ImageTk
import os
import re
import pydicom
import numpy as np
import nibabel as nib
from SAM_finetune.utils.logger_func import setup_logger

logger = setup_logger()


class ThumbnailGallery:
    """Handles the thumbnail sidebar for patient navigation and top image gallery"""
    def __init__(self, parent, sidebar_frame, on_select_image):
        self.parent = parent
        self.sidebar_frame = sidebar_frame
        self.on_select_image = on_select_image
        self.image_files = []
        self.current_image_index = -1
        self.thumbnail_refs = []  # Keep references to thumbnails
        self.patient_images = {}  # Dict to store images by patient
        self.current_patient = None  # Currently selected patient
        self.top_gallery_frame = None  # Will be created later
        
        self.patient_masses = {}  # Dict to store total mass per patient
        self.patient_mass_labels = {}  # Dict to store mass display labels
        
        # Labels for the patient list
        self.patient_label = Label(
            self.sidebar_frame, 
            text="Patients", 
            font=("Helvetica", 12, "bold"),
            bootstyle="inverse-dark"
        )
        self.patient_label.pack(side=tk.TOP, fill=tk.X, pady=3, padx=5)
        
        # Create a canvas with scrollbar for patients
        self.patient_canvas = Canvas(self.sidebar_frame, width=150)
        self.patient_scrollbar = Scrollbar(
            self.sidebar_frame, 
            orient="vertical", 
            command=self.patient_canvas.yview,
            bootstyle="dark-round"
        )
        
        self.patient_list_frame = Frame(self.patient_canvas, bootstyle="dark")
        
        self.patient_canvas.configure(yscrollcommand=self.patient_scrollbar.set)
        self.patient_canvas.create_window((0, 0), window=self.patient_list_frame, anchor="nw", width=150)
        
        self.patient_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.patient_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure the patient frame to update scroll region when size changes
        self.patient_list_frame.bind(
            "<Configure>",
            lambda e: self.patient_canvas.configure(scrollregion=self.patient_canvas.bbox("all"))
        )
        
        # Add mouse wheel scrolling
        self.patient_canvas.bind("<MouseWheel>", self._on_patient_mousewheel)
        self.patient_canvas.bind("<Button-4>", self._on_patient_mousewheel)
        self.patient_canvas.bind("<Button-5>", self._on_patient_mousewheel)
    
    def setup_top_gallery(self, control_frame):
        """Create the top gallery after control_frame is available"""
        # Create the top image gallery
        self.top_gallery_frame = Frame(control_frame, bootstyle="dark")
        self.top_gallery_frame.pack(side=tk.TOP, fill=tk.X, pady=3, padx=10)
        
        # Label for top gallery
        self.gallery_label = Label(
            self.top_gallery_frame, 
            text="Patient Images", 
            font=("Helvetica", 12, "bold"),
            bootstyle="inverse-dark"
        )
        self.gallery_label.pack(side=tk.TOP, fill=tk.X, pady=1)
        
        # Scrollable frame for top thumbnails
        self.top_canvas = Canvas(self.top_gallery_frame, height=120)
        self.top_frame = Frame(self.top_canvas, bootstyle="dark")
        self.top_scrollbar = Scrollbar(
            self.top_gallery_frame, 
            orient="horizontal", 
            command=self.top_canvas.xview,
            bootstyle="dark-round"
        )
        
        self.top_canvas.configure(xscrollcommand=self.top_scrollbar.set)
        self.top_canvas.create_window((0, 0), window=self.top_frame, anchor="nw")
        
        self.top_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.top_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.top_frame.bind(
            "<Configure>",
            lambda e: self.top_canvas.configure(scrollregion=self.top_canvas.bbox("all"))
        )
        
        self.top_canvas.bind("<MouseWheel>", self._on_top_mousewheel)
        self.top_canvas.bind("<Button-4>", self._on_top_mousewheel)
        self.top_canvas.bind("<Button-5>", self._on_top_mousewheel)
    
    def _extract_patient_id(self, filename):
        """Extract patient ID from filename or DICOM header"""
        # Get base filename without path and extension
        base_name = os.path.basename(filename)
        
        # Check if this is a DICOM file and extract from header
        if filename.endswith('.dcm') or filename.endswith('.dicom'):
            try:
                dicom_file = pydicom.dcmread(filename)
                
                # Try Patient ID first (most reliable)
                if hasattr(dicom_file, 'PatientID') and dicom_file.PatientID:
                    return str(dicom_file.PatientID).strip()
                
                # Fallback to Patient Name if no Patient ID
                if hasattr(dicom_file, 'PatientName') and dicom_file.PatientName:
                    # Convert DICOM PersonName to string
                    patient_name = str(dicom_file.PatientName).strip()
                    # Remove any special characters and replace spaces with underscores
                    patient_name = re.sub(r'[^\w\s-]', '', patient_name).replace(' ', '_')
                    return patient_name
                
                # Last resort: use Study Instance UID (truncated)
                if hasattr(dicom_file, 'StudyInstanceUID') and dicom_file.StudyInstanceUID:
                    study_uid = str(dicom_file.StudyInstanceUID)
                    # Take first part of UID for readability
                    return study_uid.split('.')[0] if '.' in study_uid else study_uid[:20]
                    
            except Exception as e:
                logger.warning(f"Could not read DICOM header from {filename}: {e}")
                # Fall back to filename parsing
        
        # Original filename parsing logic for non-DICOM files
        # Need at least 6 parts (patient ID + 5 metadata parts)
        parts = base_name.split('_')
        if len(parts) >= 6:  
            return '_'.join(parts[:-5])
        elif len(parts) == 2:
            return base_name.split('.')[0]
        
        # a_1.jpg -> a
        match = re.match(r'^([^_]+)_', base_name)
        if match:
            return match.group(1)
        
        # If no underscore, try to extract the DICOM-style ID (everything before the last numeric segment)
        parts = base_name.split('.')
        if len(parts) > 2:
            # Check if the last part is numeric (like a timestamp)
            if parts[-2].isdigit() or (len(parts[-2]) > 8 and parts[-2].isdigit()):
                # Join all parts except the last numeric segment
                return '.'.join(parts[:-2])
            else:
                # Just return everything except the file extension
                return '.'.join(parts[:-1])
            
        return None  # No patient ID found
        
    def load_images(self, image_paths):
        """Load a list of images and organize by patient"""
        if not image_paths:
            return False
        
        original_image_files = list(image_paths)
        self.image_files = []
        
        # Group images by patient ID
        self.patient_images = {}
        for image_path in original_image_files:
            patient_id = self._extract_patient_id(image_path)
            if patient_id:
                if patient_id not in self.patient_images:
                    #check nifti & add slice index
                    if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
                        # Len images 
                        slice_idx = self._len_nifti_images(image_path)
                        self.patient_images[patient_id] = []
                        for i in range(slice_idx):
                            slice_path = image_path + f'#slice={i}'
                            self.patient_images[patient_id].append(slice_path)
                            self.image_files.append(slice_path)
                    else:
                        self.patient_images[patient_id] = [image_path]
                        self.image_files.append(image_path)
                else:
                    self.patient_images[patient_id].append(image_path)
                    self.image_files.append(image_path)
            else:
                # If no patient ID could be extracted, use "Other" category
                if "Other" not in self.patient_images:
                    self.patient_images["Other"] = []
                self.patient_images["Other"].append(image_path)
        
        # Clear existing patient buttons
        for widget in self.patient_list_frame.winfo_children():
            widget.destroy()
        
        # Create patient buttons in left sidebar
        for patient_id in sorted(self.patient_images.keys()):
            img_count = len(self.patient_images[patient_id])
            patient_btn = Button(
                self.patient_list_frame,
                text=f"{patient_id} ({img_count})",
                bootstyle="primary-outline",
                width=15,
                command=lambda pid=patient_id: self.select_patient(pid)
            )
            patient_btn.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)
        
        # Select first patient if available
        if self.patient_images:
            first_patient = sorted(self.patient_images.keys())[0]
            self.select_patient(first_patient)
            return True
        return False
    
    def select_patient(self, patient_id):
        """Select a patient and display their images in the top gallery"""
        if patient_id not in self.patient_images or self.top_gallery_frame is None:
            return
        
        self.current_patient = patient_id
        patient_images = self.patient_images[patient_id]
        
        # Update patient buttons to highlight selected
        for btn in self.patient_list_frame.winfo_children():
            if patient_id in btn['text']:
                btn.configure(bootstyle="primary")
            else:
                btn.configure(bootstyle="primary-outline")
        
        # Update gallery label
        self.gallery_label.config(text=f"Patient {patient_id} Images")
        
        # Clear existing top thumbnails
        for widget in self.top_frame.winfo_children():
            widget.destroy()
        
        # Reset thumbnail references
        self.thumbnail_refs = []
        
        # Create thumbnails for patient images in top gallery
        for idx, image_path in enumerate(patient_images):
            original_image_path = image_path.split('#')[0]
            try:
                # Create frame for this thumbnail
                thumb_frame = Frame(self.top_frame, 
                                    bootstyle="dark",
                                    width=100, height=100)
                thumb_frame.pack(side=tk.LEFT, padx=5, pady=5)
                thumb_frame.pack_propagate(False)
                
                # Open and resize the image
                if original_image_path.endswith('.dcm') or original_image_path.endswith('.dicom'):
                    img = self.load_dicom_image(image_path)
                elif original_image_path.endswith('.nii') or original_image_path.endswith('.nii.gz'):
                    img = self.load_nifti_image(original_image_path)
                else:
                    img = Image.open(image_path)
                img.thumbnail((90, 90))  # Small thumbnail
                img_tk = ImageTk.PhotoImage(img)
                self.thumbnail_refs.append(img_tk)
                
                # Image label
                label = Label(thumb_frame, image=img_tk, bootstyle="dark")
                label.image = img_tk    
                label.pack(pady=(5,0))
                
                # Filename label (shortened - just show the number part)
                filename = os.path.basename(image_path)
                match = re.search(r'_(\d+)', filename)
                if match:
                    short_name = match.group(1)
                else:
                    short_name = filename[:5] + "..."
                    
                name_label = Label(thumb_frame, text=short_name, bootstyle="light", font=("Helvetica", 8))
                name_label.pack(pady=(0,5))
                
                # Bind click to select image
                def make_lambda(img_path, img_idx):
                    return lambda e: self.select_image_by_path(img_path, img_idx)
                
                thumb_frame.bind("<Button-1>", make_lambda(image_path, idx))
                label.bind("<Button-1>", make_lambda(image_path, idx))
                name_label.bind("<Button-1>", make_lambda(image_path, idx))
                
            except Exception as e:
                logger.error(f"Error creating thumbnail for {image_path}: {e}")
        
        # After creating all thumbnails, bind mousewheel to all thumbnail widgets
        for child in self.top_frame.winfo_children():
            child.bind("<MouseWheel>", self._on_top_mousewheel)
            child.bind("<Button-4>", self._on_top_mousewheel)
            child.bind("<Button-5>", self._on_top_mousewheel)
        
        # Select first image if available
        if patient_images:
            self.select_image_by_path(patient_images[0], 0)
    
    def select_image_by_path(self, image_path, local_index):
        """Select image by path and local index within patient"""
        if not image_path or self.top_gallery_frame is None:
            return
        
        # Find global index of this image
        global_index = self.image_files.index(image_path) if image_path in self.image_files else -1
        
        # Update visuals for previously selected thumbnail in top gallery
        for thumb_frame in self.top_frame.winfo_children():
            thumb_frame.configure(bootstyle="dark")
        
        # Highlight the selected thumbnail
        if local_index < len(self.top_frame.winfo_children()):
            current_frame = self.top_frame.winfo_children()[local_index]
            current_frame.configure(bootstyle="info")
        
        # Update the current index
        self.current_image_index = global_index
        
        # Call the selection callback to update main canvas
        self.on_select_image(image_path, global_index)
    
    def select_image(self, index):
        """Select image by global index"""
        if 0 <= index < len(self.image_files):
            image_path = self.image_files[index]
            
            # Find patient for this image
            for patient, images in self.patient_images.items():
                if image_path in images:
                    # Select patient first
                    if patient != self.current_patient:
                        self.select_patient(patient)
                    
                    # Then select image within patient
                    local_index = images.index(image_path)
                    self.select_image_by_path(image_path, local_index)
                    break
    
    def next_image(self):
        """Load the next image in the current patient"""
        if not self.image_files or not self.current_patient:
            return False
        
        patient_images = self.patient_images[self.current_patient]
        current_path = self.image_files[self.current_image_index] if self.current_image_index >= 0 else None
        
        if current_path in patient_images:
            local_index = patient_images.index(current_path)
            next_local_index = (local_index + 1) % len(patient_images)
            next_path = patient_images[next_local_index]
            next_global_index = self.image_files.index(next_path)
            self.select_image_by_path(next_path, next_local_index)
            return next_path, next_global_index
        else:
            # If current image not found in patient, select first image
            if patient_images:
                first_path = patient_images[0]
                first_global_index = self.image_files.index(first_path)
                self.select_image_by_path(first_path, 0)
                return first_path, first_global_index
        
        return False
    
    def prev_image(self):
        """Load the previous image in the current patient"""
        if not self.image_files or not self.current_patient:
            return False
        
        patient_images = self.patient_images[self.current_patient]
        current_path = self.image_files[self.current_image_index] if self.current_image_index >= 0 else None
        
        if current_path in patient_images:
            local_index = patient_images.index(current_path)
            prev_local_index = (local_index - 1) % len(patient_images)
            prev_path = patient_images[prev_local_index]
            prev_global_index = self.image_files.index(prev_path)
            self.select_image_by_path(prev_path, prev_local_index)
            return prev_path, prev_global_index
        else:
            # If current image not found in patient, select first image
            if patient_images:
                first_path = patient_images[0]
                first_global_index = self.image_files.index(first_path)
                self.select_image_by_path(first_path, 0)
                return first_path, first_global_index
        
        return False
    
    def _on_patient_mousewheel(self, event):
        """Handle mousewheel scrolling in the patient sidebar"""
        # Cross-platform handling of mousewheel events
        if event.num == 4 or event.delta > 0:
            # Scroll up
            self.patient_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            # Scroll down
            self.patient_canvas.yview_scroll(1, "units")
    
    def _on_top_mousewheel(self, event):
        """Handle mousewheel scrolling in the top gallery"""
        # Cross-platform handling of mousewheel events
        if event.num == 4 or event.delta > 0:
            # Scroll left
            self.top_canvas.xview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            # Scroll right
            self.top_canvas.xview_scroll(1, "units")
            
    def load_dicom_image(self, image_path):
        """Load a DICOM image from path into the canvas"""
        try:
            dicom_file = pydicom.dcmread(image_path)
            pixels = dicom_file.pixel_array
            # Normalize the pixel values to 0-255
            pixels = pixels - pixels.min()
            pixels = pixels / pixels.max() * 255
            pixels = pixels.astype(np.uint8)
            image = Image.fromarray(pixels).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading DICOM image: {e}")

    def _len_nifti_images(self, image_path):
        """Return the number of slices in a NIfTI image"""
        nifti_file = nib.load(image_path)
        return nifti_file.shape[2]

    def load_nifti_image(self, image_path, slice_idx=None):
        """Load a NIfTI image from path into the canvas"""
        try:
            nifti_file = nib.load(image_path)
            image_data = nifti_file.get_fdata()
            if slice_idx is None:
                slice_idx = 0
            slice_data = image_data[:, :, slice_idx]
            
            # Normalize the data to 0-255 range
            slice_data = ((slice_data - slice_data.min()) / 
                         (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            
            # Create grayscale image
            image = Image.fromarray(slice_data).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading NIfTI image: {e}")

    def update_patient_mass(self, patient_id, slice_mass):
        """Update the total mass for a patient"""
        if patient_id not in self.patient_masses:
            self.patient_masses[patient_id] = {}
        
        # Get the current image path
        if self.current_image_index >= 0 and self.current_image_index < len(self.image_files):
            current_path = self.image_files[self.current_image_index]
            # Update or add the mass for this slice
            self.patient_masses[patient_id][current_path] = slice_mass
            
            # Log the update for debugging
            logger.info(f"Updated mass for patient {patient_id}, image {current_path}: {slice_mass:.2f}")
            logger.info(f"Total patient mass: {sum(self.patient_masses[patient_id].values()):.2f}")
