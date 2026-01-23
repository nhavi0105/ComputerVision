# process.py
import time
import cv2
import numpy as np
import os


class ImageProcessor:
    """
    Minimal, working ImageProcessor for the Flask camera app.

    - Designed to be imported like: from process import ImageProcessor
    - Designed to be used like:
        processor = ImageProcessor()
        processed_img, results, process_time_ms = processor.process_frame(bgr_img)

    This version keeps things simple & stable:
    - (optional) save snapshot to CapturedImage/
    - grayscale
    - gaussian blur
    - (optional) canny edge
    """

    def __init__(self):
        # Placeholder fields for future steps (calibration, homography, tracking, etc.)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.homography_matrix = None

        self.previous_frame = None
        self.tracked_objects = []

        # Toggle features
        self.enable_save_snapshot = False   # set True if you want to save frames
        self.enable_edges = False           # set True if you want edge output

    # -------------------------
    # Step 1: Save image
    # -------------------------
    def capture_and_save_image(self, bgr_img, filename):
        """
        Save an image to CapturedImage/ folder.

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if bgr_img is None or not isinstance(bgr_img, np.ndarray):
                return False

            save_dir = "CapturedImage"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, filename)
            return bool(cv2.imwrite(save_path, bgr_img))
        except Exception as e:
            print("Error saving image:", e)
            return False

    # -------------------------
    # Step 1.5: Basic helpers
    # -------------------------
    def convert_to_grayscale(self, bgr_img):
        """Convert BGR image to grayscale."""
        if bgr_img is None:
            return None
        return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_filter(self, img, kernel_size=(5, 5), sigma=1.0):
        """Apply Gaussian blur."""
        if img is None:
            return None
        return cv2.GaussianBlur(img, kernel_size, sigma)

    def detect_edges_canny(self, img, threshold1=50, threshold2=150):
        """Canny edge detector (expects grayscale)."""
        if img is None:
            return None
        return cv2.Canny(img, threshold1, threshold2)

    # -------------------------
    # Main pipeline
    # -------------------------
    def process_frame(self, bgr_img):
        """
        Process a single frame.

        Returns:
            processed_img: np.ndarray (grayscale/edge image)
            results: dict
            process_time_ms: float
        """
        if bgr_img is None:
            raise ValueError("Input frame is None")

        start_time = time.perf_counter()
        results = {}

        # Optional snapshot saving (off by default)
        if self.enable_save_snapshot:
            ok = self.capture_and_save_image(bgr_img, "test_capture.jpg")
            results["saved_snapshot"] = ok

        # Grayscale
        gray = self.convert_to_grayscale(bgr_img)
        if gray is None:
            raise ValueError("Grayscale conversion failed")

        # Gaussian blur
        blurred = self.apply_gaussian_filter(gray, kernel_size=(5, 5), sigma=1.0)
        if blurred is None:
            raise ValueError("Gaussian filter failed")

        # Optional edges
        if self.enable_edges:
            edges = self.detect_edges_canny(blurred, threshold1=50, threshold2=150)
            processed_img = edges
            results["mode"] = "edges"
        else:
            processed_img = blurred
            results["mode"] = "blurred_gray"

        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        results["process_time_ms"] = round(process_time_ms, 2)

        return processed_img, results, process_time_ms

