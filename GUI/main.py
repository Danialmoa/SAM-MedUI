import tkinter as tk
from tkinter import filedialog
from ttkbootstrap import Style, Window, LabelFrame, Label, Button, Checkbutton, Frame, Canvas, Scrollbar, PanedWindow, Scale
import os
import cv2
import numpy as np
import json
from datetime import datetime
import time
from PIL import Image, ImageTk
from ttkbootstrap.tooltip import ToolTip
import pydicom
from pydicom.errors import InvalidDicomError
import csv

from model_handler import ModelHandler
from canvas_view import CanvasView
from thumbnail_gallery import ThumbnailGallery
from SAM_finetune.utils.logger_func import setup_logger
from SAM_finetune.utils.config import SAMGUIConfig

logger = setup_logger()


class SAMGUI:
    def __init__(self, root, config):
        # Set up ttk theme
        self.style = Style(theme="darkly")
        
        # Configure root window
        self.root = root
        self.root.title("SAM Segmentation Tool")
        self.root.geometry("1100x900")
        
        # Initialize components
        self.model_handler = ModelHandler(config)
        
        # Initialize variables
        self.image_path = None
        self.bbox = None
        self.point_coords = []
        self.point_labels = []
        self.drawing = False
        self.bbox_start_x = None
        self.bbox_start_y = None
        self.current_mask = None
        self.current_raw_prediction = None
        self.pixel_mass_factor = 1.0
        self.img_metadata = {}
        
        # UI state variables
        self.bbox_enabled = tk.BooleanVar(value=True)
        self.fg_points_enabled = tk.BooleanVar(value=True)
        self.mass_factor_var = tk.StringVar(value="1.0")
        self.mass_label_var = tk.StringVar(value="No segmentation")
        self.gamma_value = tk.DoubleVar(value=1.0)
        self.confidence_value = tk.DoubleVar(value=0.7)
        self.yolo_enabled = tk.BooleanVar(value=True)
        
        self.mask_temporarily_hidden = False
        self.prompts_temporarily_hidden = False
        
        # Dictionaries to store state for each image
        self.saved_masks = {}
        self.saved_prompts = {}
        
        # Create the main layout
        self.create_ui()
        
        # Set up event bindings
        self.setup_bindings()
    
    def create_ui(self):
        """Create the main UI layout and widgets"""
        # Create main paned window
        self.main_pane = PanedWindow(self.root, orient=tk.HORIZONTAL, bootstyle="dark")
        self.main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Create the sidebar for thumbnails - increase minimum width
        self.sidebar_frame = Frame(self.main_pane, bootstyle="dark", width=200)  # Increase from 150 to 200
        self.sidebar_frame.pack_propagate(False) 
        
        # Right side content
        self.right_frame = Frame(self.main_pane, bootstyle="dark")
        
        # Control frame (top of right side) - make it taller for the image gallery and three rows of controls
        self.control_frame = Frame(self.right_frame, bootstyle="dark", height=280)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Create top toolbar with navigation and actions FIRST (above images)
        self.create_toolbar()
        
        # Create the gallery handler
        self.thumbnail_gallery = ThumbnailGallery(
            self, 
            self.sidebar_frame, 
            self.on_select_image
        )
        
        # Set up the top gallery (now that control_frame exists)
        self.thumbnail_gallery.setup_top_gallery(self.control_frame)
        
        # Canvas frame (bottom of right side)
        self.canvas_frame = Frame(self.right_frame, bootstyle="dark")
        self.canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas view
        self.canvas_view = CanvasView(
            self,
            self.canvas_frame,
            self.on_mouse_down,
            self.on_mouse_move,
            self.on_mouse_up,
            self.on_zoom
        )
        
        # Add frames to PanedWindow
        self.main_pane.add(self.sidebar_frame)
        self.main_pane.add(self.right_frame)
        
        # Create status bar with multiple sections
        status_frame = Frame(self.root, bootstyle="dark")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, after=self.main_pane)
        
        # Left section for status messages
        self.status_bar = Label(
            status_frame, 
            text="Ready", 
            bootstyle="inverse",
            anchor=tk.W,
            padding=5
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Right section for zoom info
        self.zoom_status = Label(
            status_frame,
            text="Zoom: 1.0x",
            bootstyle="inverse-info",
            anchor=tk.E,
            padding=5
        )
        self.zoom_status.pack(side=tk.RIGHT, padx=5)
        
        # Add a label showing keyboard shortcuts
        self.controls_status = Label(
            status_frame,
            text="Ctrl+Wheel: Zoom | Ctrl+Drag: Pan | ←→: Navigate | Z: Hide Mask & Prompts",
            bootstyle="inverse-secondary",
            anchor=tk.E,
            padding=5
        )
        self.controls_status.pack(side=tk.RIGHT)
        
        # Crucial step: Force the UI to draw and calculate sizes
        self.root.update_idletasks() 
        
        # Delay the sash position setting to ensure proper layout
        self.root.after(100, lambda: self._set_initial_sash_position())
    
    def _set_initial_sash_position(self):
        """Set the initial sash position after UI is fully rendered"""
        try:
            # Set sash position to 200px (matching the sidebar width)
            self.main_pane.sashpos(0, 200)
            
            # Force update of thumbnail gallery layout
            if hasattr(self, 'thumbnail_gallery'):
                self.thumbnail_gallery.patient_canvas.update_idletasks()
        except Exception as e:
            logger.warning(f"Could not set initial sash position: {e}")
    
    def create_toolbar(self):
        """Create the toolbar with control buttons"""
        # Main controls frame - use standard LabelFrame (no rounded version available)
        
        toolbar = LabelFrame(self.control_frame, text="Controls", bootstyle="white")
        toolbar.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Create three separate frames for the rows
        top_row = Frame(toolbar)
        top_row.pack(fill=tk.X, expand=True, padx=5, pady=2)
        
        middle_row = Frame(toolbar)
        middle_row.pack(fill=tk.X, expand=True, padx=5, pady=2)
        
        bottom_row = Frame(toolbar)
        bottom_row.pack(fill=tk.X, expand=True, padx=5, pady=2)
        
        # TOP ROW - File and input controls
        self.load_folder_button = Button(
            top_row, 
            text="Load Folder", 
            command=self.load_folder,
            bootstyle="primary"
        )
        self.load_folder_button.pack(side=tk.LEFT, padx=2, pady=5)
        
        self.load_files_button = Button(
            top_row, 
            text="Load Files", 
            command=self.load_files,
            bootstyle="primary"
        )
        self.load_files_button.pack(side=tk.LEFT, padx=2, pady=5)
        
        # Input Mode section
        self.bbox_check = Checkbutton(
            top_row, 
            text="Bounding Box",
            variable=self.bbox_enabled,
            command=self.checkbox_changed,
            bootstyle="round-toggle"
        )
        self.bbox_check.pack(side=tk.LEFT, padx=5)

        self.fg_point_check = Checkbutton(
            top_row, 
            text="Points",
            variable=self.fg_points_enabled,
            command=self.checkbox_changed,
            bootstyle="round-toggle-success"
        )
        self.fg_point_check.pack(side=tk.LEFT, padx=5)
        
        if self.model_handler.yolo_model is not None:
            self.yolo_check = Checkbutton(
                top_row, 
                text="YOLO Auto-Detect",
                variable=self.yolo_enabled,
                bootstyle="round-toggle-warning"
            )
            self.yolo_check.pack(side=tk.LEFT, padx=5)
        
        # Add a separator
        Label(top_row, text="|", bootstyle="dark").pack(side=tk.LEFT, padx=5)
        
        # First row action buttons
        self.clear_button = Button(
            top_row, 
            text="Clear Prompts", 
            command=self.clear_prompts,
            bootstyle="secondary"
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.clear_mask_button = Button(
            top_row, 
            text="Clear Segmentation", 
            command=self.clear_mask,
            bootstyle="warning"
        )
        self.clear_mask_button.pack(side=tk.LEFT, padx=5)
        
        # MIDDLE ROW - Segmentation and saving controls
        self.segment_button = Button(
            middle_row, 
            text="Generate Segmentation", 
            command=self.generate_segmentation,
            bootstyle="success"
        )
        self.segment_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = Button(
            middle_row, 
            text="Save Mask", 
            command=self.save_mask,
            bootstyle="info"
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.save_all_button = Button(
            middle_row, 
            text="Save All Masks", 
            command=self.save_all_masks,
            bootstyle="primary"
        )
        self.save_all_button.pack(side=tk.LEFT, padx=3)
        
        # Add export button to middle row
        self.export_button = Button(
            middle_row,
            text="Export Results",
            command=self.export_results,
            bootstyle="danger"
        )
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # BOTTOM ROW - Gamma and Confidence controls
        # Add gamma correction controls to bottom row
        gamma_frame = Frame(bottom_row)
        gamma_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        Label(gamma_frame, text="Gamma:", bootstyle="white").pack(side=tk.LEFT, padx=3)
        
        gamma_slider = Scale(
            gamma_frame,
            variable=self.gamma_value,
            command=self.update_gamma,
            bootstyle="success",
            from_=0.2,
            to=1.7,
            orient=tk.HORIZONTAL,
            length=140,
            value=1.0
        )
        gamma_slider.pack(side=tk.LEFT, padx=3)
        
        # Gamma value display
        self.gamma_value_label = Label(
            gamma_frame,
            text="1.0",
            bootstyle="white",
            width=4
        )
        self.gamma_value_label.pack(side=tk.LEFT, padx=3)
        
        # Reset gamma button
        reset_gamma_btn = Button(
            gamma_frame,
            text="Reset",
            command=lambda: self.reset_gamma(),
            bootstyle="secondary",
            width=5
        )
        reset_gamma_btn.pack(side=tk.LEFT, padx=3)
        
        # Add confidence threshold controls to bottom row
        confidence_frame = Frame(bottom_row)
        confidence_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        Label(confidence_frame, text="Confidence:", bootstyle="white").pack(side=tk.LEFT, padx=3)
        
        confidence_slider = Scale(
            confidence_frame,
            variable=self.confidence_value,
            command=self.update_confidence,
            bootstyle="info",
            from_=0.3,
            to=0.99,
            orient=tk.HORIZONTAL,
            length=140,
            value=0.7
        )
        confidence_slider.pack(side=tk.LEFT, padx=3)
        
        # Confidence value display
        self.confidence_value_label = Label(
            confidence_frame,
            text="0.70",
            bootstyle="white",
            width=4
        )
        self.confidence_value_label.pack(side=tk.LEFT, padx=3)
        
        # Reset confidence button
        reset_confidence_btn = Button(
            confidence_frame,
            text="Reset",
            command=lambda: self.reset_confidence(),
            bootstyle="secondary",
            width=5
        )
        reset_confidence_btn.pack(side=tk.LEFT, padx=3)
        
        # Add tooltips to each button
        ToolTip(self.load_folder_button, text="Load multiple images from directories")
        ToolTip(self.load_files_button, text="Load individual images from files")
        ToolTip(self.bbox_check, text="Create bounding boxes around regions of interest by clicking and dragging")
        ToolTip(self.fg_point_check, text="Place foreground points by clicking on the image")
        ToolTip(self.clear_button, text="Clear all bounding boxes and points")
        ToolTip(self.clear_mask_button, text="Clear the current segmentation mask")
        ToolTip(self.segment_button, text="Generate a segmentation mask based on your input")
        ToolTip(self.save_button, text="Save the current segmentation mask")
        ToolTip(self.save_all_button, text="Save all generated segmentation masks")
        ToolTip(gamma_slider, text="Adjust gamma to enhance image visibility (values < 1.0 brighten dark areas, values > 1.0 darken bright areas)")
        ToolTip(reset_gamma_btn, text="Reset gamma to default value (1.0)")
        ToolTip(confidence_slider, text="Adjust confidence threshold for segmentation (higher values = more conservative segmentation)")
        ToolTip(reset_confidence_btn, text="Reset confidence to default value (0.7)")
        ToolTip(self.export_button, text="Export segmentation results to CSV")
    
    def suggest_bounding_box(self):
        """Use YOLO to suggest the best bounding box"""
        if self.canvas_view.original_image is None:
            self.update_status("Please load an image first")
            return
        
        if self.model_handler.yolo_model is None:
            self.update_status("YOLO model not available")
            return
        
        self.update_status("Getting YOLO suggestion...")
        
        # Get YOLO detection
        suggested_bbox = self.model_handler.detection(self.canvas_view.display_image)
        
        if suggested_bbox is not None:
            # Set the suggested bbox
            self.bbox = suggested_bbox
            self.update_status(f"YOLO suggested bbox: ({suggested_bbox[0]}, {suggested_bbox[1]}) to ({suggested_bbox[2]}, {suggested_bbox[3]})")
            
            # Redraw canvas to show the suggested bbox and automatically save it
            self.redraw_canvas()
            
            logger.info(f"YOLO suggested bounding box: {suggested_bbox}")
            logger.info(f"Saved bbox to prompts for image: {self.image_path}")
        else:
            self.update_status("YOLO could not detect any objects in the image")
    
    def setup_bindings(self):
        """Set up keyboard and other bindings"""
        # Add keyboard bindings for zoom
        self.root.bind("<Control-plus>", lambda e: self.on_zoom_key(1))
        self.root.bind("<Control-minus>", lambda e: self.on_zoom_key(-1))
        self.root.bind("<Control-0>", lambda e: self.reset_zoom())
        
        # Add keyboard for navigation
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        
        # Add keyboard bindings for pan
        self.root.bind("<Control-Left>", lambda e: self.pan_view(-20, 0))
        self.root.bind("<Control-Right>", lambda e: self.pan_view(20, 0))
        self.root.bind("<Control-Up>", lambda e: self.pan_view(0, -20))
        self.root.bind("<Control-Down>", lambda e: self.pan_view(0, 20))
        
        # Add panning with mouse
        self.canvas_view.canvas.bind("<Control-ButtonPress-1>", self.on_pan_start)
        self.canvas_view.canvas.bind("<Control-B1-Motion>", self.on_pan_move)
        self.canvas_view.canvas.bind("<Control-ButtonRelease-1>", self.on_pan_end)
        
        # mask and prompts toggle binding
        self.root.bind("<KeyPress-z>", self.hide_mask_and_prompts)
        self.root.bind("<KeyRelease-z>", self.show_mask_and_prompts)
    
    # Image loading and navigation methods
    def load_folder(self):
        """Load images from selected directories"""
        all_image_paths = []
        
        # Get directories
        directories = self._select_multiple_directories()
        
        if not directories:
            return
            
        # User selected directories - load all images from them
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.dcm', '.dicom', '.nii', '.nii.gz')
        
        total_files = 0
        for directory in directories:
            dir_files = 0
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(supported_extensions):
                        full_path = os.path.join(root, file)
                        all_image_paths.append(full_path)
                        dir_files += 1
            total_files += dir_files
            logger.info(f"Found {dir_files} image files in {directory}")
        
        if all_image_paths:
            self.update_status(f"Loading {total_files} images from {len(directories)} directories...")
            # Load the images into the thumbnail gallery
            if self.thumbnail_gallery.load_images(all_image_paths):
                self.update_status(f"Successfully loaded {len(all_image_paths)} images from {len(directories)} directories")
            else:
                self.update_status("Failed to load images")
        else:
            self.update_status("No supported image files found in selected directories")

    def load_files(self):
        """Load individual image files"""
        # Individual file selection
        image_paths = filedialog.askopenfilenames(
            title="Select image files",
            filetypes=[("All supported", "*.jpg *.jpeg *.png *.bmp *.dcm *.dicom *.nii"),
                       ("JPEG files", "*.jpg *.jpeg"),
                       ("PNG files", "*.png"),
                       ("BMP files", "*.bmp"),
                       ("DICOM files", "*.dcm *.dicom"),
                       ("NIfTI files", "*.nii"),
                       ("All files", "*.*")]
        )
        
        if not image_paths:
            return
        
        all_image_paths = list(image_paths)
        self.update_status(f"Loading {len(all_image_paths)} selected files...")
        
        # Load the images into the thumbnail gallery
        if self.thumbnail_gallery.load_images(all_image_paths):
            self.update_status(f"Successfully loaded {len(all_image_paths)} individual files")
        else:
            self.update_status("Failed to load images")

    def _select_multiple_directories(self):
        """Allow user to select multiple directories"""
        directories = []
        

        directory = filedialog.askdirectory(
            title="Select directory with images (Cancel to select individual files instead)"
        )
        if not directory:
            return
        directories.append(directory)
            
        return directories
    
    def on_select_image(self, image_path, index):
        """Called when an image is selected from the gallery"""
        # Save current state before changing image
        if self.image_path:
            self.save_current_state()
        
        # Load the selected image
        self.image_path = image_path
        self.original_image_path = image_path.split('#')[0]
        img_data = self.canvas_view.load_image(image_path)
        
        if img_data:
            if self.original_image_path.endswith('.nii') or self.original_image_path.endswith('.nii.gz'):
                self.img_metadata = self.extract_nifti_metadata(img_data)
            else:
                self.img_metadata = self.extract_dicom_metadata(img_data)
            self.update_pixel_mass_factor(self.img_metadata['pixel_spacing'], self.img_metadata['slice_thickness'])
        else:
            self.img_metadata = {}
            self.pixel_mass_factor = 1.0 
            logger.info("Loaded a non-DICOM image or DICOM metadata not fully available for mass calculation. Defaulting pixel_mass_factor to 1.0.")    
        
        # Reset view when changing images
        self.reset_zoom()
        
        # Update window title
        image_name = os.path.basename(image_path)
        self.root.title(f"SAM Segmentation Tool - {image_name} ({index + 1}/{len(self.thumbnail_gallery.image_files)})")
        
        # Restore previously saved state for this image
        self.restore_saved_state()
        
        # Update status
        self.update_status(f"Loaded image: {image_name}")
    
    def next_image(self):
        """Load the next image in the gallery"""
        result = self.thumbnail_gallery.next_image()
        if result:
            image_path, index = result
            self.on_select_image(image_path, index)
    
    def prev_image(self):
        """Load the previous image in the gallery"""
        result = self.thumbnail_gallery.prev_image()
        if result:
            image_path, index = result
            self.on_select_image(image_path, index)
    
    # State management methods
    def save_current_state(self):
        """Save current mask and prompts for the current image"""
        if self.image_path:
            # Save mask if it exists
            if self.current_mask is not None:
                self.saved_masks[self.image_path] = self.current_mask.copy()
                self.canvas_view.current_mask = self.current_mask.copy()
            
            # Save prompts (bounding box and points)
            self.saved_prompts[self.image_path] = {
                'bbox': self.bbox.copy() if self.bbox else None,
                'points': self.point_coords.copy() if self.point_coords else [],
                'labels': self.point_labels.copy() if self.point_labels else []
            }
    
    def restore_saved_state(self):
        """Restore saved mask and prompts for the current image"""
        # Save the freshly detected YOLO bbox (if any) before clearing
        fresh_yolo_bbox = self.bbox
        
        # Clear current state
        self.current_mask = None
        self.canvas_view.current_mask = None
        self.current_raw_prediction = None
        self.bbox = None
        self.point_coords = []
        self.point_labels = []

        # Restore mask if available
        if self.image_path in self.saved_masks:
            self.current_mask = self.saved_masks[self.image_path].copy()
            self.canvas_view.current_mask = self.current_mask.copy()
        
        # Restore prompts if available
        if self.image_path in self.saved_prompts:
            saved_state = self.saved_prompts[self.image_path]
            
            if saved_state['bbox']:
                self.bbox = saved_state['bbox'].copy()
            
            if saved_state['points']:
                self.point_coords = saved_state['points'].copy()
                self.point_labels = saved_state['labels'].copy()
        
        # If no saved bbox exists AND we have a fresh YOLO detection AND YOLO is enabled
        # use the fresh YOLO bbox (this prevents carrying over old bboxes while allowing new detections)
        if self.bbox is None and fresh_yolo_bbox is not None and self.yolo_enabled.get():
            self.bbox = fresh_yolo_bbox
            logger.info(f"Using fresh YOLO detected bbox: {fresh_yolo_bbox}")
        
        # Redraw canvas with restored state
        self.redraw_canvas()
    
    # Canvas interaction methods
    def on_mouse_down(self, event):
        if self.canvas_view.original_image is None:
            return
        
        # If control is pressed, we're panning, not editing
        if event.state & 0x4:  # Control key is pressed
            return
            
        # Convert from screen coordinates to image coordinates
        img_x, img_y = self.canvas_view.screen_to_image_coords(event.x, event.y)
        
        # Check if we're clicking near a bbox handle
        if self.bbox:
            handle = self.canvas_view.is_near_bbox_handle(event.x, event.y, self.bbox)
            if handle:
                self.canvas_view.is_editing_bbox = True
                self.canvas_view.edit_handle = handle
                return
        
        # If not editing bbox, proceed with normal point/bbox creation
        self.bbox_start_x = img_x
        self.bbox_start_y = img_y
        self.drawing = True
        self.redraw_canvas()

    def on_mouse_move(self, event):
        if self.canvas_view.original_image is None:
            return
            
        # If control is pressed, we're panning
        if event.state & 0x4:
            return
            
        # Handle bbox editing
        if self.canvas_view.is_editing_bbox:
            self.bbox = self.canvas_view.update_bbox_on_drag(event.x, event.y, self.bbox)
            self.redraw_canvas()
            return
            
        # Handle normal drawing
        if self.drawing and self.bbox_enabled.get():
            # Convert from screen coordinates to image coordinates
            img_x, img_y = self.canvas_view.screen_to_image_coords(event.x, event.y)
            
            # For display purposes only (temporary rectangle during dragging)
            temp_image = self.canvas_view.displayed_image.copy()
            
            # Calculate the starting point in displayed coordinates
            start_screen_x = self.bbox_start_x * self.canvas_view.zoom_level
            start_screen_y = self.bbox_start_y * self.canvas_view.zoom_level
            
            if self.canvas_view.zoom_level > 1.0:
                # Adjust for the current view position
                start_screen_x -= self.canvas_view.current_view_start_x
                start_screen_y -= self.canvas_view.current_view_start_y
            
            # Draw temporary rectangle in screen coordinates
            cv2.rectangle(temp_image, 
                         (int(start_screen_x), int(start_screen_y)), 
                         (event.x, event.y), 
                         (255, 0, 0), 2)
            
            self.canvas_view.image_tk = ImageTk.PhotoImage(Image.fromarray(temp_image))
            self.canvas_view.canvas.itemconfig(self.canvas_view.canvas_image, image=self.canvas_view.image_tk)
            self.canvas_view.canvas.image = self.canvas_view.image_tk

    def on_mouse_up(self, event):
        if self.canvas_view.original_image is None:
            return
        
        # If we were editing the bbox
        if self.canvas_view.is_editing_bbox:
            self.canvas_view.is_editing_bbox = False
            self.canvas_view.edit_handle = None
            self.redraw_canvas()
            return
            
        # If control is pressed, we're panning, not creating points/boxes
        if event.state & 0x4:  # Control key is pressed
            return
        
        # If we were in drawing mode
        if self.drawing:
            self.drawing = False
            
            # Convert from screen coordinates to image coordinates
            img_x, img_y = self.canvas_view.screen_to_image_coords(event.x, event.y)
            
            # Calculate distance moved during click-drag (in unzoomed coordinates)
            move_distance = ((img_x - self.bbox_start_x)**2 + 
                           (img_y - self.bbox_start_y)**2)**0.5
            
            # If this was just a click (minimal movement) and points are enabled, add a point
            if move_distance < 5 and self.fg_points_enabled.get():
                point_label = 1  # Foreground point
                # Store original (unzoomed) coordinates
                self.point_coords.append([img_x, img_y])
                self.point_labels.append(point_label)
                
                # Log point selection
                image_name = os.path.basename(self.image_path) if self.image_path else "unknown"
                logger.info(f"User added point at ({img_x}, {img_y}) on image: {image_name}")
                self.update_status(f"Added foreground point at ({img_x}, {img_y})")
            
            # If this was a drag and bbox is enabled, create bounding box
            elif move_distance >= 5 and self.bbox_enabled.get():
                x1 = min(self.bbox_start_x, img_x)
                y1 = min(self.bbox_start_y, img_y)
                x2 = max(self.bbox_start_x, img_x)
                y2 = max(self.bbox_start_y, img_y)
                
                if x1 != x2 and y1 != y2:
                    self.bbox = [x1, y1, x2, y2]
                    
                    # Log bounding box creation
                    image_name = os.path.basename(self.image_path) if self.image_path else "unknown"
                    logger.info(f"User created bounding box [{x1},{y1},{x2},{y2}] on image: {image_name}")
                    self.update_status(f"Created bounding box ({x2-x1}x{y2-y1})")
        
        self.redraw_canvas()
    
    def checkbox_changed(self):
        # We no longer need to enforce exclusivity between checkboxes
        pass
    
    # Zoom and pan methods
    def on_zoom(self, event):
        """Handle zooming with mouse wheel"""
        if self.canvas_view.displayed_image is None:
            return
        
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            # Zoom in
            new_zoom = min(5.0, self.canvas_view.zoom_level * 1.1)
        elif event.num == 5 or event.delta < 0:
            # Zoom out
            new_zoom = max(0.5, self.canvas_view.zoom_level / 1.1)
        else:
            return
        
        self.apply_zoom(new_zoom)
    
    def on_zoom_key(self, direction):
        """Handle zooming with keyboard shortcuts"""
        if self.canvas_view.displayed_image is None:
            return
        
        if direction > 0:
            # Zoom in
            new_zoom = min(5.0, self.canvas_view.zoom_level * 1.1)
        else:
            # Zoom out
            new_zoom = max(0.5, self.canvas_view.zoom_level / 1.1)
        
        self.apply_zoom(new_zoom)
    
    def reset_zoom(self, event=None):
        """Reset zoom level to 1.0 and reset panning"""
        if self.canvas_view.displayed_image is None:
            return
        
        self.canvas_view.view_offset_x = 0
        self.canvas_view.view_offset_y = 0
        self.apply_zoom(1.0)

    def apply_zoom(self, new_zoom):
        """Apply the given zoom level"""
        # Only update if zoom actually changed
        if self.canvas_view.zoom_level != new_zoom:
            self.canvas_view.zoom_level = new_zoom
            logger.info(f"Zoom level changed to {new_zoom:.2f}x")
            self.redraw_canvas()
            
            # Update zoom status in the status bar only
            self.zoom_status.configure(text=f"Zoom: {new_zoom:.1f}x")
    
    def on_pan_start(self, event):
        """Start panning the view"""
        if self.canvas_view.displayed_image is None or self.canvas_view.zoom_level <= 1.0:
            return
        
        self.canvas_view.is_panning = True
        self.canvas_view.pan_start_x = event.x
        self.canvas_view.pan_start_y = event.y
    
    def on_pan_move(self, event):
        """Pan the view as the mouse moves"""
        if not self.canvas_view.is_panning or self.canvas_view.displayed_image is None:
            return
        
        # Calculate how much to move
        dx = event.x - self.canvas_view.pan_start_x
        dy = event.y - self.canvas_view.pan_start_y
        
        # Update the view offset
        self.pan_view(dx, dy)
        
        # Update starting position for next move
        self.canvas_view.pan_start_x = event.x
        self.canvas_view.pan_start_y = event.y
    
    def on_pan_end(self, event):
        """End panning"""
        self.canvas_view.is_panning = False
    
    def pan_view(self, dx, dy):
        """Pan the view by dx, dy pixels"""
        if self.canvas_view.displayed_image is None or self.canvas_view.zoom_level <= 1.0:
            return
        
        # Update the view offset
        self.canvas_view.view_offset_x += dx
        self.canvas_view.view_offset_y += dy
        
        # Calculate max allowed offset based on image size and zoom level
        zoomed_width = int(self.canvas_view.display_image.shape[1] * self.canvas_view.zoom_level)
        zoomed_height = int(self.canvas_view.display_image.shape[0] * self.canvas_view.zoom_level)
        
        view_width = self.canvas_view.display_image.shape[1]
        view_height = self.canvas_view.display_image.shape[0]
        
        max_offset_x = zoomed_width - view_width
        max_offset_y = zoomed_height - view_height
        
        # Constrain the offset to valid ranges
        self.canvas_view.view_offset_x = max(-max_offset_x, min(self.canvas_view.view_offset_x, 0))
        self.canvas_view.view_offset_y = max(-max_offset_y, min(self.canvas_view.view_offset_y, 0))
        
        # Important: Force redraw to update the current_view_start values
        self.redraw_canvas()
    
    # Segmentation methods
    def generate_segmentation(self):
        """Generate a segmentation mask based on the current prompts"""
        if self.canvas_view.original_image is None:
            self.update_status("Please load an image first")
            return
        
        if self.bbox is None and not self.point_coords:
            self.update_status("Please provide at least one bounding box or point")
            return
        
        segmentation_start_time = time.time()
        image_name = os.path.basename(self.image_path)
        logger.info(f"Starting segmentation for image: {image_name}")
        logger.info(f"Prompt data - Bounding box: {self.bbox}, Point count: {len(self.point_coords)}")
        
        self.update_status("Generating segmentation...")
        
        # Use the model handler to generate the mask
        cropped_image, resize_factor = self._crop_image_for_segmentation(self.canvas_view.display_image)
        
        # Adjust prompts for the cropped image
        cropped_bbox = None
        cropped_points = []
        cropped_labels = []
        
        # Original image dimensions
        orig_h, orig_w = self.canvas_view.display_image.shape[:2]
        # Cropped image dimensions
        crop_size = cropped_image.shape[0]  # Square image
        
        # Calculate the offset from original to cropped coordinates
        x_offset = (orig_w - crop_size) // 2
        y_offset = (orig_h - crop_size) // 2
        
        # Adjust bounding box for cropped image
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            cropped_bbox = [
                max(0, min(crop_size, x1 - x_offset)),
                max(0, min(crop_size, y1 - y_offset)),
                max(0, min(crop_size, x2 - x_offset)),
                max(0, min(crop_size, y2 - y_offset))
            ]
        
        # Adjust points for cropped image
        for i, (x, y) in enumerate(self.point_coords):
            # Convert to cropped coordinates
            cx = x - x_offset
            cy = y - y_offset
            
            # Only include points that fall within the cropped area
            if 0 <= cx < crop_size and 0 <= cy < crop_size:
                cropped_points.append([cx, cy])
                cropped_labels.append(self.point_labels[i])
        
        # Generate mask on the cropped image
        cropped_mask, raw_prediction = self.model_handler.generate_mask(
            cropped_image,
            bbox=cropped_bbox,
            points=cropped_points,
            point_labels=cropped_labels,
            confidence_threshold=self.confidence_value.get()
        )
        
        cropped_mask = cv2.resize(cropped_mask, (crop_size, crop_size))
        
        if cropped_mask is None:
            self.update_status("Segmentation failed")
            return
        
        # Store the raw prediction for threshold adjustments
        raw_prediction_resized = cv2.resize(raw_prediction, (crop_size, crop_size))
        
        # Convert cropped mask back to original image size
        full_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
        full_raw_prediction = np.zeros((orig_h, orig_w), dtype=np.float32)
        
        # Place the cropped mask and raw prediction in the center of the full arrays
        full_mask[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size] = cropped_mask
        full_raw_prediction[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size] = raw_prediction_resized
        
        self.current_mask = full_mask
        self.current_raw_prediction = full_raw_prediction  # Store raw prediction
        self.canvas_view.current_mask = full_mask
        self.canvas_view.current_resize_factor = resize_factor
        
        # Save the mask for this image
        self.saved_masks[self.image_path] = self.current_mask.copy()
        self.saved_prompts[self.image_path] = {
            'bbox': self.bbox.copy() if self.bbox else None,
            'points': self.point_coords.copy() if self.point_coords else [],
            'labels': self.point_labels.copy() if self.point_labels else []
        }
        
        # Calculate and display segment stats
        pixel_count, mass = self.canvas_view.update_stats_overlay(resize_factor)
        
        # After calculating mass in the existing code:
        if self.thumbnail_gallery.current_patient:
            self.thumbnail_gallery.update_patient_mass(self.thumbnail_gallery.current_patient, mass)
        
        segmentation_time = time.time() - segmentation_start_time
        logger.info(f"Segmentation complete. Time taken: {segmentation_time:.2f} seconds")
        self.update_status(f"Segmentation complete: {pixel_count} pixels, {mass:.2f} mass")
        
        self.redraw_canvas()
    
    def _crop_image_for_segmentation(self, segmentation_image):
        """Crop the image minimum dimension"""
        min_dim = min(segmentation_image.shape[0], segmentation_image.shape[1])
        center = np.array(segmentation_image.shape) // 2
        cropped_image = segmentation_image[center[0]-min_dim//2:center[0]+min_dim//2, center[1]-min_dim//2:center[1]+min_dim//2]

        original_min_dim = min(self.canvas_view.original_image.shape[0], self.canvas_view.original_image.shape[1])
        
        resize_factor = 1024.0 / original_min_dim
        return cropped_image, resize_factor
    
    def clear_prompts(self):
        """Clear all prompts (bounding box and points)"""
        self.bbox = None
        self.point_coords = []
        self.point_labels = []
        self.drawing = False
        
        if self.canvas_view.displayed_image is not None:
            self.redraw_canvas()
            self.update_status("Cleared all prompts")
    
    def clear_mask(self):
        """Clear only the generated segmentation mask, keeping prompts"""
        if self.current_mask is not None:
            if self.thumbnail_gallery.current_patient:
                current_patient = self.thumbnail_gallery.current_patient
                if current_patient in self.thumbnail_gallery.patient_masses:
                    # Remove this image's mass from the patient masses dictionary
                    if self.image_path in self.thumbnail_gallery.patient_masses[current_patient]:
                        del self.thumbnail_gallery.patient_masses[current_patient][self.image_path]
                        logger.info(f"Removed mass for image {self.image_path} from patient {current_patient}")
            
            self.current_mask = None
            self.current_raw_prediction = None
            self.canvas_view.current_mask = None
            
            # Also remove from saved masks
            if self.image_path in self.saved_masks:
                del self.saved_masks[self.image_path]
            
            self.canvas_view.update_stats_overlay()
            self.redraw_canvas()
            self.update_status("Segmentation mask cleared")
            self.mass_label_var.set("No segmentation")
    
    def redraw_canvas(self):
        """Redraw the canvas with current state and save prompts"""
        # Save current bbox and points to saved_prompts if we have an image loaded
        if self.image_path:
            self.saved_prompts[self.image_path] = {
                'bbox': self.bbox.copy() if self.bbox else None,
                'points': self.point_coords.copy() if self.point_coords else [],
                'labels': self.point_labels.copy() if self.point_labels else []
            }
        
        # Determine what to show based on temporarily hidden flags
        show_bbox = self.bbox if not self.prompts_temporarily_hidden else None
        show_points = self.point_coords if not self.prompts_temporarily_hidden else []
        show_labels = self.point_labels if not self.prompts_temporarily_hidden else []
        
        # Redraw the canvas with current state
        self.canvas_view.draw_image_with_annotations(
            bbox=show_bbox,
            point_coords=show_points,
            point_labels=show_labels,
            gamma=self.gamma_value.get(),
            mask_visible=not self.mask_temporarily_hidden
        )
    
    # File saving methods
    def save_mask(self):
        """Save the current segmentation mask to a file"""
        if self.current_mask is None:
            self.update_status("No mask generated yet")
            return
        
        # Create a "masks" and "prompts" directories if they don't exist
        masks_dir = os.path.join(os.path.dirname(self.image_path), "masks")
        os.makedirs(masks_dir, exist_ok=True)
        prompts_dir = os.path.join(os.path.dirname(self.image_path), "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        
        # Generate mask filename based on original image name
        image_name = os.path.basename(self.image_path)
        base_name, ext = os.path.splitext(image_name)
        
        if '#slice=' in self.image_path:
            slice_idx = self.image_path.split('#slice=')[1]
            base_name = f"{base_name}_slice{slice_idx}"
        
        mask_path = os.path.join(masks_dir, f"{base_name}_mask.png")
        prompt_path = os.path.join(prompts_dir, f"{base_name}_prompt.json")
        
        # Option to override default path
        suggested_path = mask_path
        save_path = filedialog.asksaveasfilename(
            initialfile=os.path.basename(suggested_path),
            initialdir=os.path.dirname(suggested_path),
            defaultextension=".png", 
            filetypes=[("PNG files", "*.png")]
        )
        
        if not save_path:
            return
        
        mask_image = (self.current_mask * 255).astype(np.uint8)
        try:
            cv2.imwrite(save_path, mask_image)
            self.update_status(f"Mask saved to {save_path}")
            
            if self.image_path in self.saved_prompts:
                prompt_data = self.saved_prompts[self.image_path]
                
                # Convert to serializable format
                bbox = prompt_data['bbox'].tolist() if isinstance(prompt_data['bbox'], np.ndarray) else prompt_data['bbox']
                
                points = []
                for i, (x, y) in enumerate(prompt_data['points']):
                    if i < len(prompt_data['labels']):
                        label = prompt_data['labels'][i]
                        point_type = "positive" if label == 1 else "negative"
                        points.append({
                            "coordinates": [float(x), float(y)],
                            "type": point_type
                        })
                
                # Create JSON structure
                json_data = {
                    "image": {
                        "name": image_name,
                        "path": self.image_path
                    },
                    "prompt": {
                        "bounding_box": bbox,
                        "points": points,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "mask": {
                        "path": save_path
                    }
                }
                
                # Add stats
                pixel_count = np.sum(self.current_mask)
                mass = pixel_count * self.canvas_view.pixel_mass_factor
                json_data["mask"]["stats"] = {
                    "pixel_count": int(pixel_count),
                    "mass_factor": float(self.canvas_view.pixel_mass_factor),
                    "calculated_mass": float(mass)
                }
                
                with open(prompt_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                logger.info(f"Saved mask to {save_path} and prompt data to {prompt_path}")
        except Exception as e:
            logger.error(f"Error saving mask: {e}")
            self.update_status(f"Error saving mask: {e}")
    
    def save_all_masks(self):
        """Save all generated masks to a directory"""
        if not self.saved_masks:
            self.update_status("No masks have been generated yet")
            return
        
        # Ask user to select a directory to save all masks
        base_dir = filedialog.askdirectory(title="Select directory to save all masks")
        if not base_dir:
            return
        
        # Create directories
        masks_dir = os.path.join(base_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        prompts_dir = os.path.join(base_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        
        # Save each mask
        saved_count = 0
        for img_path, mask in self.saved_masks.items():
            # Generate filename based on original image name
            original_path = img_path.split('#')[0] if '#' in img_path else img_path
            image_name = os.path.basename(original_path)
            base_name, ext = os.path.splitext(image_name)
            
            if '#slice=' in img_path:
                slice_idx = img_path.split('#slice=')[1]
                base_name = f"{base_name}_slice{slice_idx}"
            
            mask_path = os.path.join(masks_dir, f"{base_name}_mask.png")
            prompt_path = os.path.join(prompts_dir, f"{base_name}_prompt.json")
            
            # Save the mask
            mask_image = (mask * 255).astype(np.uint8)
            try:
                cv2.imwrite(mask_path, mask_image)
                saved_count += 1
                
                if img_path in self.saved_prompts:
                    prompt_data = self.saved_prompts[img_path]
                    
                    # Convert to serializable format
                    bbox = prompt_data['bbox'].tolist() if isinstance(prompt_data['bbox'], np.ndarray) else prompt_data['bbox']
                    
                    points = []
                    for i, (x, y) in enumerate(prompt_data['points']):
                        if i < len(prompt_data['labels']):
                            label = prompt_data['labels'][i]
                            point_type = "foreground" if label == 1 else "background"
                            points.append({
                                "coordinates": [float(x), float(y)],
                                "type": point_type
                            })
                    
                    # Create JSON structure for individual prompt
                    json_data = {
                        "image": {
                            "name": image_name,
                            "path": img_path
                        },
                        "prompt": {
                            "bounding_box": bbox,
                            "points": points,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        },
                        "mask": {
                            "path": mask_path
                        }
                    }
                    
                    # Add stats
                    pixel_count = np.sum(mask)
                    mass = pixel_count * self.canvas_view.pixel_mass_factor
                    json_data["mask"]["stats"] = {
                        "pixel_count": int(pixel_count),
                        "mass_factor": float(self.canvas_view.pixel_mass_factor),
                        "calculated_mass": float(mass)
                    }
                    
                    # Save individual prompt JSON
                    with open(prompt_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                    
                    logger.info(f"Saved mask and prompt for {image_name}")
            except Exception as e:
                logger.error(f"Error saving mask for {image_name}: {e}")
        
        # Save all prompts to a consolidated file
        all_prompts_path = os.path.join(prompts_dir, "all_prompts.json")
        try:
            with open(all_prompts_path, "w") as f:
                json.dump(self.saved_prompts, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving all prompts: {e}")
        
        # Show confirmation message
        if saved_count > 0:
            logger.info(f"Successfully saved {saved_count} masks to {masks_dir}")
            self.update_status(f"Saved {saved_count} masks to {masks_dir}")
        else:
            logger.error("No masks were saved")
            self.update_status("No masks were saved")
    
    # Utility methods
    def update_status(self, message):
        """Update the status bar with a message"""
        self.status_bar.configure(text=message)
        logger.info(message)

    def update_gamma(self, value=None):
        """Update gamma value and redraw canvas"""
        # Update the value label
        self.gamma_value_label.configure(text=f"{self.gamma_value.get():.2f}")
        # Update the image with new gamma
        self.redraw_canvas()
        self.update_status(f"Gamma: {self.gamma_value.get():.2f}")

    def reset_gamma(self):
        """Reset gamma to default (1.0)"""
        self.gamma_value.set(1.0)
        self.update_gamma()

    def update_confidence(self, value=None):
        """Update confidence threshold value and reapply to existing prediction"""
        # Update the value label
        self.confidence_value_label.configure(text=f"{self.confidence_value.get():.2f}")
        
        if self.current_raw_prediction is not None:
            # Reapply threshold to existing raw prediction
            new_mask = self.model_handler.apply_confidence_threshold(
                self.current_raw_prediction, 
                self.confidence_value.get()
            )
            
            # Update current mask
            self.current_mask = new_mask.astype(np.float32)
            self.canvas_view.current_mask = self.current_mask
            
            # Update saved mask for this image
            if self.image_path:
                self.saved_masks[self.image_path] = self.current_mask.copy()
            
            # Recalculate stats and redraw
            pixel_count, mass = self.canvas_view.update_stats_overlay()
            if self.thumbnail_gallery.current_patient and mass is not None:
                self.thumbnail_gallery.update_patient_mass(self.thumbnail_gallery.current_patient, mass)
            
            self.redraw_canvas()
            self.update_status(f"Confidence threshold: {self.confidence_value.get():.2f}")
        else:
            self.update_status(f"Confidence threshold: {self.confidence_value.get():.2f} (no active segmentation)")

    def reset_confidence(self):
        """Reset confidence to default (0.7)"""
        self.confidence_value.set(0.7)
        self.update_confidence()

    def extract_dicom_metadata(self, dicom_data):
        """Extract relevant metadata from a DICOM dataset."""
        metadata = {
            'patient_id': str(dicom_data.get('PatientID', 'N/A')),
            'study_date': str(dicom_data.get('StudyDate', 'N/A')),
            'modality': str(dicom_data.get('Modality', 'N/A')),
            'pixel_spacing': None,
            'slice_thickness': None,
            'window_center': 'N/A',
            'window_width': 'N/A',
            # Add more tags as needed: e.g., dicom_data.get('Manufacturer', 'N/A')
        }
        if hasattr(dicom_data, 'PixelSpacing') and dicom_data.PixelSpacing:
            try:
                metadata['pixel_spacing'] = [float(x) for x in dicom_data.PixelSpacing]
            except TypeError:
                metadata['pixel_spacing'] = [float(dicom_data.PixelSpacing)]


        if hasattr(dicom_data, 'SliceThickness') and dicom_data.SliceThickness is not None:
            try:
                metadata['slice_thickness'] = float(dicom_data.SliceThickness)
            except (TypeError, ValueError):
                metadata['slice_thickness'] = 'N/A' # Or handle appropriately


        wc = dicom_data.get('WindowCenter', None)
        ww = dicom_data.get('WindowWidth', None)

        if wc is not None:
            try:
                metadata['window_center'] = [float(x) for x in wc] if isinstance(wc, pydicom.multival.MultiValue) else float(wc)
            except (TypeError, ValueError):
                 metadata['window_center'] = 'N/A'
        
        if ww is not None:
            try:
                metadata['window_width'] = [float(x) for x in ww] if isinstance(ww, pydicom.multival.MultiValue) else float(ww)
            except (TypeError, ValueError):
                metadata['window_width'] = 'N/A'
                
        metadata['pixel_spacing'] = [float(x) for x in dicom_data.PixelSpacing]
        metadata['slice_thickness'] = float(dicom_data.SliceThickness)
            
        return metadata

    def update_pixel_mass_factor(self, pixel_spacing, slice_thickness):
        """Update pixel mass factor based on metadata"""
        self.canvas_view.pixel_mass_factor = 1.0
        ps_x, ps_y, st = None, None, None
        try:
            ps_x, ps_y = float(pixel_spacing[0]), float(pixel_spacing[1])
            st = float(slice_thickness)
        except (TypeError, ValueError):
            logger.warning("Invalid metadata for pixel spacing or slice thickness. Using default values.")
            return
        
        # Calculate the pixel mass factor based on the pixel spacing and slice thickness
        self.canvas_view.pixel_mass_factor = (ps_x * ps_y * st) / 1000 * 1.05 # 1.05 is a density factor
        
        logger.info(f"Updated pixel mass factor to {self.canvas_view.pixel_mass_factor} based on metadata.")
            
    def extract_nifti_metadata(self, nifti_data):
        """Extract relevant metadata from a NIfTI dataset."""
        metadata = {
            'patient_id': str(nifti_data.header.get('PatientID', 'N/A')),
            'study_date': str(nifti_data.header.get('StudyDate', 'N/A')),
            'modality': str(nifti_data.header.get('Modality', 'N/A')),
        }
        pixel_dims = nifti_data.header['pixdim'][1:4]
        metadata['pixel_spacing'] = [float(pixel_dims[0]), float(pixel_dims[1])]
        metadata['slice_thickness'] = float(pixel_dims[2])
        return metadata
    
    def hide_mask_and_prompts(self, event):
        """Temporarily hide the mask and prompts when key is pressed"""
        self.mask_temporarily_hidden = True
        self.prompts_temporarily_hidden = True
        self.redraw_canvas()
            
    def show_mask_and_prompts(self, event):
        """Show the mask and prompts again when key is released"""
        self.mask_temporarily_hidden = False
        self.prompts_temporarily_hidden = False
        self.redraw_canvas()

    def export_results(self):
        """Export segmentation results to CSV"""
        if not self.thumbnail_gallery.patient_masses:
            self.update_status("No segmentation results to export")
            return
        
        # Ask user where to save the CSV
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"segmentation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['Patient ID', 'Image Name', 'Slice', 'Mass'])
                
                # Write data for each patient
                for patient_id, slice_masses in self.thumbnail_gallery.patient_masses.items():
                    for image_path, mass in slice_masses.items():
                        # Extract image name and slice info
                        base_name = os.path.basename(image_path.split('#')[0])
                        slice_info = "N/A"
                        if '#slice=' in image_path:
                            slice_info = image_path.split('#slice=')[1]
                        
                        writer.writerow([
                            patient_id,
                            base_name,
                            slice_info,
                            f"{mass:.2f}"
                        ])
        
            self.update_status(f"Results exported to {file_path}")
            logger.info(f"Segmentation results exported to {file_path}")
        except Exception as e:
            self.update_status(f"Error exporting results: {str(e)}")
            logger.error(f"Error exporting results: {e}")

if __name__ == "__main__":
    config = SAMGUIConfig(
        model_type='vit_b',
        sam_path='checkpoints/sam_vit_b_01ec64.pth',
        checkpoint_path='checkpoints/best_model.pth',
        device='cpu',
        yolo_model_path='checkpoints/yolo_best.pt',
    )

    root = Window(themename="darkly")
    app = SAMGUI(root, config)
    root.mainloop()