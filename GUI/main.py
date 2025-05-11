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

from model_handler import ModelHandler
from canvas_view import CanvasView
from thumbnail_gallery import ThumbnailGallery
from SAM_finetune.utils.logger_func import setup_logger

logger = setup_logger()


class SAMGUI:
    def __init__(self, root, config):
        # Set up ttk theme
        self.style = Style(theme="darkly")
        
        # Configure root window
        self.root = root
        self.root.title("SAM Segmentation Tool")
        self.root.geometry("800x900")
        
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
        self.pixel_mass_factor = 1.0
        self.img_metadata = {}
        
        # UI state variables
        self.bbox_enabled = tk.BooleanVar(value=True)
        self.fg_points_enabled = tk.BooleanVar(value=True)
        self.mass_factor_var = tk.StringVar(value="1.0")
        self.mass_label_var = tk.StringVar(value="No segmentation")
        self.gamma_value = tk.DoubleVar(value=1.0)
        
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
        
        # Create the sidebar for thumbnails
        self.sidebar_frame = Frame(self.main_pane, bootstyle="dark")
        
        # Right side content
        self.right_frame = Frame(self.main_pane, bootstyle="dark")
        
        # Control frame (top of right side) - make it taller for the image gallery
        self.control_frame = Frame(self.right_frame, bootstyle="dark", height=220)
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
            text="Ctrl+Wheel: Zoom | Ctrl+Drag: Pan | ←→: Navigate",
            bootstyle="inverse-secondary",
            anchor=tk.E,
            padding=5
        )
        self.controls_status.pack(side=tk.RIGHT)
        
        # Crucial step: Force the UI to draw and calculate sizes
        self.root.update_idletasks() 

        self.main_pane.sashpos(0, 150)
    
    def create_toolbar(self):
        """Create the toolbar with control buttons"""
        # Main controls frame - use standard LabelFrame (no rounded version available)
        
        toolbar = LabelFrame(self.control_frame, text="Controls", bootstyle="white")
        toolbar.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Create two separate frames for the rows
        top_row = Frame(toolbar)
        top_row.pack(fill=tk.X, expand=True, padx=5, pady=2)
        
        bottom_row = Frame(toolbar)
        bottom_row.pack(fill=tk.X, expand=True, padx=5, pady=2)
        
        # TOP ROW - File and input controls
        self.load_button = Button(
            top_row, 
            text="Load Images", 
            command=self.load_images,
            bootstyle="primary"
        )
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        
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
        
        # BOTTOM ROW - Segmentation and saving controls
        self.segment_button = Button(
            bottom_row, 
            text="Generate Segmentation", 
            command=self.generate_segmentation,
            bootstyle="success"
        )
        self.segment_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = Button(
            bottom_row, 
            text="Save Mask", 
            command=self.save_mask,
            bootstyle="info"
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.save_all_button = Button(
            bottom_row, 
            text="Save All Masks", 
            command=self.save_all_masks,
            bootstyle="primary"
        )
        self.save_all_button.pack(side=tk.LEFT, padx=3)
        
        # Add a separator
        Label(bottom_row, text="|", bootstyle="secondary").pack(side=tk.LEFT, padx=5)
        
        # Add gamma correction controls to bottom row
        gamma_frame = Frame(bottom_row)
        gamma_frame.pack(side=tk.LEFT, padx=3, fill=tk.Y)
        
        Label(gamma_frame, text="Gamma:", bootstyle="white").pack(side=tk.LEFT, padx=3)
        
        gamma_slider = Scale(
            gamma_frame,
            variable=self.gamma_value,
            command=self.update_gamma,
            bootstyle="success",
            from_=0.5,
            to=1.5,
            orient=tk.HORIZONTAL,
            length=120,
            value=1.0
        )
        gamma_slider.pack(side=tk.LEFT, padx=3)
        
        # Reset gamma button
        reset_gamma_btn = Button(
            gamma_frame,
            text="Reset",
            command=lambda: self.reset_gamma(),
            bootstyle="secondary",
            width=5
        )
        reset_gamma_btn.pack(side=tk.LEFT, padx=3)
        
        # Add tooltips to each button
        ToolTip(self.load_button, text="Load medical images for segmentation")
        ToolTip(self.bbox_check, text="Create bounding boxes around regions of interest by clicking and dragging")
        ToolTip(self.fg_point_check, text="Place foreground points by clicking on the image")
        ToolTip(self.clear_button, text="Clear all bounding boxes and points")
        ToolTip(self.clear_mask_button, text="Clear the current segmentation mask")
        ToolTip(self.segment_button, text="Generate a segmentation mask based on your input")
        ToolTip(self.save_button, text="Save the current segmentation mask")
        ToolTip(self.save_all_button, text="Save all generated segmentation masks")
        ToolTip(gamma_slider, text="Adjust gamma to enhance image visibility (values < 1.0 brighten dark areas, values > 1.0 darken bright areas)")
        ToolTip(reset_gamma_btn, text="Reset gamma to default value (1.0)")
    
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
    
    # Image loading and navigation methods
    def load_images(self):
        """Allow selecting multiple images and store them for navigation"""
        image_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                       ("DICOM files", "*.dcm *.dicom"),
                       ("NIfTI files", "*.nii"),
                       ("All files", "*.*")]
        )
        
        if not image_paths:
            return
        
        # Load the images into the thumbnail gallery
        if self.thumbnail_gallery.load_images(image_paths):
            self.update_status("Loaded images successfully")
        else:
            self.update_status("Failed to load images")
    
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
        # Clear current state
        self.current_mask = None
        self.canvas_view.current_mask = None
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
        
        # Redraw canvas with restored state
        self.redraw_canvas()
    
    # Canvas interaction methods
    def on_mouse_down(self, event):
        if self.canvas_view.original_image is None:
            return
        
        # If control is pressed, we're panning, not creating points/boxes
        if event.state & 0x4:  # Control key is pressed
            return
        
        # Start user interaction timer
        self.interaction_start_time = time.time()
        image_name = os.path.basename(self.image_path) if self.image_path else "unknown"
        logger.info(f"User started interaction on image: {image_name}")
        
        # Convert from screen coordinates to image coordinates
        img_x, img_y = self.canvas_view.screen_to_image_coords(event.x, event.y)
        
        # Store initial click position for both potential point and bbox
        self.bbox_start_x = img_x
        self.bbox_start_y = img_y
        
        # Always set drawing mode to true to handle both points and bboxes
        self.drawing = True
        
        self.redraw_canvas()
    
    def on_mouse_move(self, event):
        if self.canvas_view.original_image is None or not self.drawing or not self.bbox_enabled.get():
            return
        
        # If control is pressed, we're panning, not creating points/boxes
        if event.state & 0x4:  # Control key is pressed
            return
        
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
        
        # Log interaction duration
        if hasattr(self, 'interaction_start_time'):
            interaction_time = time.time() - self.interaction_start_time
            image_name = os.path.basename(self.image_path) if self.image_path else "unknown"
            logger.info(f"User interaction completed in {interaction_time:.2f} seconds on image: {image_name}")
        
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
        self.current_mask = self.model_handler.generate_mask(
            self.canvas_view.display_image,
            bbox=self.bbox,
            points=self.point_coords,
            point_labels=self.point_labels
        )
        
        if self.current_mask is None:
            self.update_status("Segmentation failed")
            return
        
        # Update canvas view with the new mask
        self.canvas_view.current_mask = self.current_mask
        
        # Save the mask for this image
        self.saved_masks[self.image_path] = self.current_mask.copy()
        self.saved_prompts[self.image_path] = {
            'bbox': self.bbox.copy() if self.bbox else None,
            'points': self.point_coords.copy() if self.point_coords else [],
            'labels': self.point_labels.copy() if self.point_labels else []
        }
        
        # Calculate and display segment stats
        pixel_count, mass = self.canvas_view.update_stats_overlay()
        
        segmentation_time = time.time() - segmentation_start_time
        logger.info(f"Segmentation complete. Time taken: {segmentation_time:.2f} seconds")
        self.update_status(f"Segmentation complete: {pixel_count} pixels, {mass:.2f} mass")
        
        self.redraw_canvas()
    
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
            self.current_mask = None
            self.canvas_view.current_mask = None
            
            # Also remove from saved masks
            if self.image_path in self.saved_masks:
                del self.saved_masks[self.image_path]
            
            self.redraw_canvas()
            self.update_status("Segmentation mask cleared")
            self.mass_label_var.set("No segmentation")
    
    def redraw_canvas(self):
        """Redraw the canvas with current state"""
        self.canvas_view.draw_image_with_annotations(
            bbox=self.bbox,
            point_coords=self.point_coords,
            point_labels=self.point_labels,
            gamma=self.gamma_value.get()
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
        # Update the image with new gamma
        self.redraw_canvas()
        self.update_status(f"Gamma: {self.gamma_value.get():.2f}")

    def reset_gamma(self):
        """Reset gamma to default (1.0)"""
        self.gamma_value.set(1.0)
        self.update_gamma()

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

if __name__ == "__main__":
    config = {
        'model_type': 'vit_b',
        'sam_path': 'checkpoints/sam_vit_b_01ec64.pth',
        'checkpoint_path': 'checkpoints/best_model.pth',
        'device': 'mps'
    }

    root = Window(themename="darkly")
    app = SAMGUI(root, config)
    root.mainloop()