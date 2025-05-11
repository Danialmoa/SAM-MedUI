import tkinter as tk
from ttkbootstrap import Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import logging
import time
import pydicom

from SAM_finetune.utils.logger_func import setup_logger

logger = setup_logger()

class CanvasView:
    """Handles the image canvas and drawing operations"""
    def __init__(self, parent, canvas_frame, on_mouse_down, on_mouse_move, on_mouse_up, on_zoom):
        self.parent = parent
        self.canvas = Canvas(canvas_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
        # Initialize image related variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.displayed_image = None
        self.current_mask = None
        self.image_tk = None
        self.canvas_image = None
        self.pixel_mass_factor = 1.0 
        
        # Initialize zoom related variables
        self.zoom_level = 1.0
        self.zoom_text_id = None
        self.help_text_id = None
        self.stats_text_id = None
        
        # Initialize panning variables
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.current_view_start_x = 0
        self.current_view_start_y = 0
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", on_mouse_down)
        self.canvas.bind("<B1-Motion>", on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", on_mouse_up)
        self.canvas.bind("<Control-MouseWheel>", on_zoom)
        self.canvas.bind("<Control-Button-4>", on_zoom)
        self.canvas.bind("<Control-Button-5>", on_zoom)

    def load_image(self, image_path):
        """Load an image from path into the canvas"""
        if image_path.endswith('.dcm') or image_path.endswith('.dicom'):
            self.original_image, dicom_data = self.load_dicom_image(image_path)
        else:
            original_img = Image.open(image_path).convert('RGB')
            self.original_image = np.array(original_img)
            dicom_data = None
        
        h_orig, w_orig = self.original_image.shape[:2]

        # Get current canvas dimensions.
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        scale_factor = 1.0

        # Calculate scale factor to fit original image into canvas dimensions
        scale_factor_w = canvas_w / w_orig
        scale_factor_h = canvas_h / h_orig
        scale_factor = min(scale_factor_w, scale_factor_h)

        # Calculate new dimensions
        new_w = int(w_orig * scale_factor)
        new_h = int(h_orig * scale_factor)

        # Ensure dimensions are at least 1x1 to avoid errors with cv2.resize
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # Resize image for display
        self.display_image = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.displayed_image = self.display_image.copy()

        # Reset view parameters
        self.reset_view()
        
        # Update canvas
        self.update_canvas()
        
        return dicom_data
    
    def reset_view(self):
        """Reset zoom and pan settings"""
        self.zoom_level = 1.0
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.current_view_start_x = 0
        self.current_view_start_y = 0
        self.clear_overlays()
    
    def clear_overlays(self):
        """Clear all text overlays from canvas"""
        if self.stats_text_id:
            self.canvas.delete(self.stats_text_id)
            self.canvas.delete("stats_bg")
            self.stats_text_id = None
        
        if self.zoom_text_id:
            self.canvas.delete(self.zoom_text_id)
            self.canvas.delete("zoom_bg")
            self.zoom_text_id = None
        
        if self.help_text_id:
            self.canvas.delete(self.help_text_id)
            self.canvas.delete("help_bg")
            self.help_text_id = None
    
    def update_canvas(self):
        """Update the canvas with the current image"""
        if self.displayed_image is None:
            return
        
        # Convert to PIL Image and then to PhotoImage for tkinter
        self.image_tk = ImageTk.PhotoImage(Image.fromarray(self.displayed_image))
        
        # Display image on canvas
        if self.canvas_image:
            self.canvas.itemconfig(self.canvas_image, image=self.image_tk)
            self.canvas.image = self.image_tk
        else:
            self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            self.canvas.image = self.image_tk
        
        # Draw overlays
        self.update_stats_overlay()
    
    def update_stats_overlay(self):
        """Update the stats overlay with mask information"""
        # Remove previous stats text if it exists
        if self.stats_text_id:
            self.canvas.delete(self.stats_text_id)
            self.canvas.delete("stats_bg")
            self.stats_text_id = None
        
        # If no mask, don't display stats
        if self.current_mask is None:
            return None, None
        
        # Calculate stats
        pixel_count = np.sum(self.current_mask)
        mass = pixel_count * self.pixel_mass_factor
        
        # Create text for overlay
        stats_text = f"Pixels: {pixel_count:,}\nMass: {mass:.2f}"
        
        # Add text overlay to canvas
        self.stats_text_id = self.canvas.create_text(
            20, 20,  # Position
            text=stats_text,
            anchor=tk.NW,
            fill="#ffffff",  # Using explicit hex code for white
            font=("Helvetica", 14, "bold")
        )
        
        # Add a semi-transparent background rectangle
        bbox = self.canvas.bbox(self.stats_text_id)
        if bbox:
            padding = 10
            rect_id = self.canvas.create_rectangle(
                bbox[0]-padding, bbox[1]-padding, 
                bbox[2]+padding, bbox[3]+padding,
                fill='black', outline='',
                stipple='gray50',  # Makes it semi-transparent
                tags="stats_bg"
            )
            self.canvas.tag_lower(rect_id, self.stats_text_id)  # Put rectangle behind text
        
        # Log the information
        image_name = os.path.basename(self.image_path) if self.image_path else "unknown"
        logger.info(f"Segment stats for {image_name}: {pixel_count} pixels, mass = {mass:.2f}")
        
        return pixel_count, mass

    
    def draw_image_with_annotations(self, bbox=None, point_coords=None, point_labels=None, gamma=1.0):
        """Redraw the image with annotations (bbox and points) and apply gamma correction"""
        if self.display_image is None:
            return
        
        # Start with a clean copy of the display image
        orig_h, orig_w = self.display_image.shape[:2]
        
        # Apply gamma correction to display image before zoom
        display_with_gamma = self.display_image.copy()
        if gamma != 1.0:
            display_with_gamma = self.apply_gamma(display_with_gamma, gamma)
        
        # Apply zoom to the entire image first
        if self.zoom_level != 1.0:
            # Calculate new dimensions
            new_h, new_w = int(orig_h * self.zoom_level), int(orig_w * self.zoom_level)
            
            # Resize the displayed image with zoom
            zoomed_image = cv2.resize(
                display_with_gamma, 
                (new_w, new_h), 
                interpolation=cv2.INTER_LINEAR if self.zoom_level > 1.0 else cv2.INTER_AREA
            )
            
            # Determine the visible region after panning
            if self.zoom_level > 1.0:
                # Calculate the viewport size (same as original image size)
                view_h, view_w = orig_h, orig_w
                
                # Calculate where to start the view based on pan offset
                start_x = max(0, min(-self.view_offset_x, new_w - view_w))
                start_y = max(0, min(-self.view_offset_y, new_h - view_h))
                
                # Ensure we don't try to view outside the image bounds
                end_x = min(start_x + view_w, new_w)
                end_y = min(start_y + view_h, new_h)
                
                # Extract the viewable portion
                self.displayed_image = zoomed_image[start_y:end_y, start_x:end_x].copy()
                
                # Store the view offset for coordinate transformations
                self.current_view_start_x = start_x
                self.current_view_start_y = start_y
            else:
                self.displayed_image = zoomed_image
                self.current_view_start_x = 0
                self.current_view_start_y = 0
        else:
            self.displayed_image = display_with_gamma
            self.current_view_start_x = 0
            self.current_view_start_y = 0
        
        # Now overlay the mask if it exists
        if self.current_mask is not None:
            # Resize mask to match the current view
            display_h, display_w = self.displayed_image.shape[:2]
            
            # First zoom the mask to match the zoomed image
            zoomed_mask = cv2.resize(
                self.current_mask.astype(np.uint8), 
                (int(orig_w * self.zoom_level), int(orig_h * self.zoom_level)),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Then crop to match the current view if we're zoomed in
            if self.zoom_level > 1.0:
                # Extract the viewable portion of the mask
                start_x = self.current_view_start_x
                start_y = self.current_view_start_y
                end_x = start_x + display_w
                end_y = start_y + display_h
                
                # Ensure we don't exceed bounds
                end_x = min(end_x, zoomed_mask.shape[1])
                end_y = min(end_y, zoomed_mask.shape[0])
                
                scaled_mask = zoomed_mask[start_y:end_y, start_x:end_x]
            else:
                # No cropping needed for 1.0 zoom
                scaled_mask = zoomed_mask
            
            # Create overlay
            mask_overlay = np.zeros_like(self.displayed_image)
            
            # Ensure the scaled_mask dimensions match the displayed_image
            if scaled_mask.shape[:2] != self.displayed_image.shape[:2]:
                # This can happen if there's a rounding issue - resize to match exactly
                scaled_mask = cv2.resize(
                    scaled_mask,
                    (self.displayed_image.shape[1], self.displayed_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            mask_overlay[scaled_mask > 0] = [0, 255, 0]  # Green for the mask
            alpha = 0.5
            self.displayed_image = cv2.addWeighted(self.displayed_image, 1, mask_overlay, alpha, 0)
        
        # Draw bounding box if it exists
        if bbox:
            # Apply zoom and pan to the bbox
            x1, y1, x2, y2 = bbox
            
            # First apply zoom
            x1 = int(x1 * self.zoom_level)
            y1 = int(y1 * self.zoom_level)
            x2 = int(x2 * self.zoom_level)
            y2 = int(y2 * self.zoom_level)
            
            # Then adjust for pan offset
            if self.zoom_level > 1.0:
                x1 -= self.current_view_start_x
                y1 -= self.current_view_start_y
                x2 -= self.current_view_start_x
                y2 -= self.current_view_start_y
            
            # Check if bounding box is still in view
            h, w = self.displayed_image.shape[:2]
            if x1 < w and x2 > 0 and y1 < h and y2 > 0:
                # Clip to display area
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                # Draw only if we have valid coordinates
                if x1 < x2 and y1 < y2:
                    cv2.rectangle(self.displayed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw points
        if point_coords and point_labels:
            for i, (x, y) in enumerate(point_coords):
                # Apply zoom 
                x_zoomed = int(x * self.zoom_level)
                y_zoomed = int(y * self.zoom_level)
                
                # Adjust for pan offset
                if self.zoom_level > 1.0:
                    x_zoomed -= self.current_view_start_x
                    y_zoomed -= self.current_view_start_y
                
                # Check if point is in view
                h, w = self.displayed_image.shape[:2]
                if 0 <= x_zoomed < w and 0 <= y_zoomed < h:
                    label = point_labels[i] if i < len(point_labels) else 1
                    color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green=FG, Red=BG
                    cv2.circle(self.displayed_image, (x_zoomed, y_zoomed), 5, color, -1)
        
        self.update_canvas()
    
    def screen_to_image_coords(self, screen_x, screen_y):
        """Convert screen coordinates to original image coordinates"""
        # When zoomed in, we need to account for the current view position
        if self.zoom_level > 1.0:
            # Add current_view_start_x/y to get coordinates in the full zoomed image
            adjusted_x = screen_x + self.current_view_start_x
            adjusted_y = screen_y + self.current_view_start_y
        else:
            adjusted_x = screen_x
            adjusted_y = screen_y
        
        # Then convert from zoomed coordinates to original image coordinates
        img_x = int(adjusted_x / self.zoom_level)
        img_y = int(adjusted_y / self.zoom_level)
        
        return img_x, img_y

    def apply_gamma(self, image, gamma):
        """Apply gamma correction to an image"""
        if gamma == 1.0:
            return image.copy()  # No change at gamma 1.0
        
        # OpenCV expects gamma value in a different way, so we need to use the inverse
        inv_gamma = 1.0 / gamma
        
        # Create a lookup table
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply gamma correction using the lookup table
        return cv2.LUT(image.copy(), table)
    
    def load_dicom_image(self, image_path):
        """Load a DICOM image from path into the canvas"""
        try:
            dicom_file = pydicom.dcmread(image_path)
            pixels = dicom_file.pixel_array
            image = Image.fromarray(pixels).convert('RGB')
            image_array = np.array(image)
            return image_array, dicom_file
        except Exception as e:
            logger.error(f"Error loading DICOM image: {e}")
            return None, None