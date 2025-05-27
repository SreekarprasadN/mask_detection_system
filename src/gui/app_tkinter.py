"""
Lightweight facial mask detection system using OpenCV only.
This version doesn't require TensorFlow or other ML libraries.
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import threading

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)  # Add project root to Python path

# Configure logging with robust path handling
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mask_detection_app')

class FaceDetector:
    """Face detector class using OpenCV's built-in face detector."""
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the face detector.
        
        Args:
            confidence_threshold (float): Minimum confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_dir = os.path.join(project_root, 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Paths to the model files
        prototxt_path = os.path.join(model_dir, 'deploy.prototxt')
        caffemodel_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        
        # Download model files if they don't exist
        self._ensure_model_files(prototxt_path, caffemodel_path)
        
        # Load the face detector model
        self.net = cv2.dnn.readNet(prototxt_path, caffemodel_path)
        
        # Check if OpenCL is available and set preferableTarget
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            logger.info("Using OpenCL for face detection")
        else:
            logger.info("OpenCL not available, using CPU for face detection")
        
        # Initialize the mask classifier
        self.mask_classifier = MaskClassifier()
        
        logger.info("Face detector initialized successfully")
    
    def _ensure_model_files(self, prototxt_path, caffemodel_path):
        """
        Ensure model files exist, download if they don't.
        
        Args:
            prototxt_path (str): Path to the prototxt file
            caffemodel_path (str): Path to the caffemodel file
        """
        import urllib.request
        
        # URLs for the model files
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        
        # Download prototxt if needed
        if not os.path.exists(prototxt_path):
            logger.info(f"Downloading face detector prototxt to {prototxt_path}")
            try:
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
            except Exception as e:
                logger.error(f"Failed to download prototxt: {str(e)}")
                raise
        
        # Download caffemodel if needed
        if not os.path.exists(caffemodel_path):
            logger.info(f"Downloading face detector model to {caffemodel_path}")
            try:
                urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
            except Exception as e:
                logger.error(f"Failed to download caffemodel: {str(e)}")
                raise
    
    def detect_faces(self, frame):
        """
        Detect faces in an image frame.
        
        Args:
            frame (numpy.ndarray): Input image frame
            
        Returns:
            list: List of face bounding boxes (x, y, w, h)
        """
        if frame is None:
            logger.error("Input frame is None")
            return []
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        
        # Set the input to the network
        self.net.setInput(blob)
        
        # Forward pass to get detections
        detections = self.net.forward()
        
        # List to store face bounding boxes
        face_boxes = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")
                
                # Ensure the bounding box is within the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Calculate width and height
                width = endX - startX
                height = endY - startY
                
                # Add to face boxes if valid dimensions
                if width > 0 and height > 0:
                    face_boxes.append((startX, startY, width, height))
        
        return face_boxes
    
    def extract_face_roi(self, frame, face_box, target_size=(224, 224)):
        """
        Extract and preprocess a face region from the frame.
        
        Args:
            frame (numpy.ndarray): Input image frame
            face_box (tuple): Face bounding box (x, y, w, h)
            target_size (tuple): Target size for the extracted face
            
        Returns:
            numpy.ndarray: Preprocessed face image
        """
        x, y, w, h = face_box
        
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Resize to target size
        face_roi = cv2.resize(face_roi, target_size)
        
        # Convert to RGB (from BGR)
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        return face_roi
    
    def detect_mask(self, frame, face_box):
        """
        Detect if a face is wearing a mask.
        
        Args:
            frame (numpy.ndarray): Input image frame
            face_box (tuple): Face bounding box (x, y, w, h)
            
        Returns:
            tuple: (has_mask, confidence)
        """
        # Extract face ROI
        face_roi = self.extract_face_roi(frame, face_box)
        
        # Use the mask classifier to predict
        return self.mask_classifier.predict(face_roi)

class MaskClassifier:
    """Simple mask classifier using color and edge features."""
    
    def __init__(self):
        """Initialize the mask classifier."""
        # No initialization needed for this simple classifier
        pass
    
    def predict(self, face_image):
        """
        Predict whether a face image contains a mask.
        
        Args:
            face_image (numpy.ndarray): Face image in RGB format
            
        Returns:
            tuple: (has_mask, confidence)
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
        
        # Define range for common mask colors (light blue, light green, white)
        lower_mask = np.array([90, 50, 50])
        upper_mask = np.array([130, 255, 255])
        
        # Create mask for blue/green colors
        mask1 = cv2.inRange(hsv, lower_mask, upper_mask)
        
        # Define range for white masks
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white colors
        mask2 = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate the percentage of mask pixels in the lower half of the face
        h, w = face_image.shape[:2]
        lower_half = combined_mask[h//2:, :]
        mask_pixel_percentage = np.sum(lower_half > 0) / (lower_half.size)
        
        # Calculate edge density in the lower face region
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray[h//2:, :], 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine features
        # If high mask pixel percentage and low edge density, likely wearing mask
        if mask_pixel_percentage > 0.15 and edge_density < 0.1:
            has_mask = True
            confidence = min(0.95, mask_pixel_percentage * 2)
        else:
            has_mask = False
            confidence = min(0.95, (1 - mask_pixel_percentage) * 1.5)
        
        return (has_mask, confidence)

class MaskDetectionApp:
    """Main application window for mask detection using Tkinter."""
    
    def __init__(self, root):
        """
        Initialize the application.
        
        Args:
            root: Tkinter root window
        """
        # Initialize variables
        self.root = root
        self.root.title("Facial Mask Detection System")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        self.face_detector = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.log_detections = True
        self.save_violations = False
        self.detection_threshold = 0.7
        self.camera_id = 0
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        
        # Set up the UI
        self.init_ui()
        
        # Initialize face detector
        self.face_detector = FaceDetector(confidence_threshold=0.5)
        
        logger.info("Application initialized successfully")
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create video display frame
        self.video_frame = ttk.Frame(main_frame, borderwidth=2, relief="groove")
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create video canvas
        self.video_canvas = tk.Canvas(self.video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create status display
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            font=("Arial", 24, "bold"),
            background="#2196F3",
            foreground="white",
            padding=10,
            anchor="center"
        )
        self.status_label.pack(fill=tk.X, pady=10)
        
        # Create controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Create grid layout for controls
        for i in range(4):
            controls_frame.columnconfigure(i, weight=1)
        
        # Camera selection
        camera_frame = ttk.LabelFrame(controls_frame, text="Camera")
        camera_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.camera_var = tk.IntVar(value=0)
        ttk.Radiobutton(camera_frame, text="Default Camera", variable=self.camera_var, value=0).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(camera_frame, text="Camera 1", variable=self.camera_var, value=1).pack(anchor="w", padx=5, pady=2)
        ttk.Button(camera_frame, text="Apply", command=self.change_camera).pack(anchor="w", padx=5, pady=5)
        
        # Threshold control
        threshold_frame = ttk.LabelFrame(controls_frame, text="Detection Threshold")
        threshold_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.threshold_var = tk.DoubleVar(value=0.7)
        threshold_scale = ttk.Scale(
            threshold_frame, 
            from_=0.5, 
            to=0.9, 
            orient="horizontal", 
            variable=self.threshold_var,
            command=lambda v: self.threshold_value_label.config(text=f"{float(v):.1f}")
        )
        threshold_scale.pack(fill=tk.X, padx=5, pady=5)
        self.threshold_value_label = ttk.Label(threshold_frame, text="0.7")
        self.threshold_value_label.pack(padx=5)
        ttk.Button(threshold_frame, text="Apply", command=self.change_threshold).pack(padx=5, pady=5)
        
        # Logging options
        log_frame = ttk.LabelFrame(controls_frame, text="Logging Options")
        log_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        
        self.log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_frame, text="Log Detections", variable=self.log_var, command=self.toggle_logging).pack(anchor="w", padx=5, pady=2)
        
        self.save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(log_frame, text="Save Violations", variable=self.save_var, command=self.toggle_save_violations).pack(anchor="w", padx=5, pady=2)
        
        # Control buttons
        button_frame = ttk.LabelFrame(controls_frame, text="Controls")
        button_frame.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_video)
        self.start_button.pack(fill=tk.X, padx=5, pady=2)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_video, state="disabled")
        self.stop_button.pack(fill=tk.X, padx=5, pady=2)
        
        # Status bar
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        fps_label = ttk.Label(status_bar, textvariable=self.fps_var)
        fps_label.pack(side=tk.RIGHT, padx=10)
        
        # Set initial status
        self.update_status("Ready", "normal")
    
    def start_video(self):
        """Start video capture and processing."""
        if self.is_running:
            return
        
        try:
            # Open camera
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {self.camera_id}")
            
            # Update UI
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # Start processing
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.process_video)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            self.update_status("Monitoring...", "normal")
            logger.info("Video capture started")
            
        except Exception as e:
            logger.error(f"Error starting video: {str(e)}")
            messagebox.showerror("Error", f"Failed to start video: {str(e)}")
    
    def stop_video(self):
        """Stop video capture and processing."""
        if not self.is_running:
            return
        
        # Stop processing
        self.is_running = False
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Update UI
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
        self.update_status("Stopped", "normal")
        logger.info("Video capture stopped")
    
    def process_video(self):
        """Process video frames in a separate thread."""
        while self.is_running:
            try:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Update FPS calculation
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                
                if elapsed_time >= 1.0:  # Update FPS every second
                    self.fps = self.frame_count / elapsed_time
                    self.root.after(0, lambda: self.fps_var.set(f"FPS: {self.fps:.1f}"))
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update display
                self.current_frame = processed_frame
                self.root.after(0, self.update_display)
                
                # Sleep to control frame rate
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error processing video frame: {str(e)}")
        
        logger.info("Video processing stopped")
    
    def process_frame(self, frame):
        """
        Process a video frame for mask detection.
        
        Args:
            frame (numpy.ndarray): Input video frame
            
        Returns:
            numpy.ndarray: Processed frame with detection results
        """
        # Make a copy of the frame
        result_frame = frame.copy()
        
        # Detect faces
        face_boxes = self.face_detector.detect_faces(frame)
        
        # Process each detected face
        for face_box in face_boxes:
            x, y, w, h = face_box
            
            # Detect mask
            has_mask, confidence = self.face_detector.detect_mask(frame, face_box)
            
            # Determine status based on prediction
            if has_mask:  # With mask
                if confidence >= self.detection_threshold:
                    status = "Safe - Mask detected"
                    color = (0, 255, 0)  # Green
                    self.root.after(0, lambda s=status: self.update_status(s, "safe"))
                    self.log_detection(status, confidence, face_box)
                else:
                    status = "Uncertain"
                    color = (255, 165, 0)  # Orange
                    self.root.after(0, lambda s=status: self.update_status(s, "uncertain"))
            else:  # Without mask
                if confidence >= self.detection_threshold:
                    status = "No mask - Not safe"
                    color = (0, 0, 255)  # Red
                    self.root.after(0, lambda s=status: self.update_status(s, "unsafe"))
                    self.log_detection(status, confidence, face_box)
                    
                    # Save violation image if enabled
                    if self.save_violations:
                        self.save_violation_image(frame, face_box)
                else:
                    status = "Uncertain"
                    color = (255, 165, 0)  # Orange
                    self.root.after(0, lambda s=status: self.update_status(s, "uncertain"))
            
            # Draw bounding box and status
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_frame, f"{status} ({confidence:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_frame
    
    def update_display(self):
        """Update the video display with the current frame."""
        if self.current_frame is None:
            return
        
        # Convert to RGB for Tkinter
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has valid dimensions
            h, w = frame_rgb.shape[:2]
            
            # Calculate aspect ratio
            aspect_ratio = w / h
            
            # Calculate new dimensions
            if canvas_width / canvas_height > aspect_ratio:
                # Canvas is wider than frame
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Canvas is taller than frame
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)
            
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            
            # Update canvas
            self.video_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.photo, anchor=tk.CENTER
            )
    
    def update_status(self, text, status_type):
        """
        Update the status display.
        
        Args:
            text (str): Status text
            status_type (str): Status type ('safe', 'unsafe', 'uncertain', or 'normal')
        """
        self.status_var.set(text)
        
        if status_type == "safe":
            self.status_label.config(background="#4CAF50")  # Green
        elif status_type == "unsafe":
            self.status_label.config(background="#F44336")  # Red
        elif status_type == "uncertain":
            self.status_label.config(background="#FF9800")  # Orange
        else:
            self.status_label.config(background="#2196F3")  # Blue
    
    def log_detection(self, status, confidence, face_box):
        """
        Log detection events.
        
        Args:
            status (str): Detection status
            confidence (float): Confidence score
            face_box (tuple): Face bounding box (x, y, w, h)
        """
        if not self.log_detections:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        x, y, w, h = face_box
        log_entry = f"{timestamp} - {status} - Confidence: {confidence:.2f} - Position: ({x}, {y}, {w}, {h})"
        
        logger.info(log_entry)
        
        # Write to log file with robust path handling
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'detections.log')
        
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def save_violation_image(self, frame, face_box):
        """
        Save an image of a mask violation.
        
        Args:
            frame (numpy.ndarray): Current video frame
            face_box (tuple): Face bounding box (x, y, w, h)
        """
        # Create violations directory with robust path handling
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        violations_dir = os.path.join(project_root, 'logs', 'violations')
        os.makedirs(violations_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(violations_dir, f"violation_{timestamp}.jpg")
        
        # Save the image
        cv2.imwrite(filename, frame)
        logger.info(f"Violation image saved: {filename}")
    
    def change_camera(self):
        """Change the camera source."""
        new_camera_id = self.camera_var.get()
        
        if new_camera_id != self.camera_id:
            self.camera_id = new_camera_id
            logger.info(f"Camera changed to {self.camera_id}")
            
            # Restart video if running
            if self.is_running:
                self.stop_video()
                self.start_video()
    
    def change_threshold(self):
        """Change the detection threshold."""
        self.detection_threshold = self.threshold_var.get()
        logger.info(f"Detection threshold changed to {self.detection_threshold}")
    
    def toggle_logging(self):
        """Toggle detection logging."""
        self.log_detections = self.log_var.get()
        logger.info(f"Detection logging {'enabled' if self.log_detections else 'disabled'}")
    
    def toggle_save_violations(self):
        """Toggle saving violation images."""
        self.save_violations = self.save_var.get()
        logger.info(f"Saving violation images {'enabled' if self.save_violations else 'disabled'}")
    
    def on_closing(self):
        """Handle window close event."""
        # Stop video if running
        if self.is_running:
            self.stop_video()
        
        # Destroy window
        self.root.destroy()

def main():
    """Main function to run the application."""
    # Create root window
    root = tk.Tk()
    
    # Create application
    app = MaskDetectionApp(root)
    
    # Set window close handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Configure window resizing
    def on_resize(event):
        if event.widget == root:
            app.update_display()
    
    root.bind("<Configure>", on_resize)
    
    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()
