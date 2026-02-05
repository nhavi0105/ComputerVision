import time
from Week1_Capturering.Week1_captureSaveImg import CaptureSaveImgProcessor

from Week2_Filtering.Week2_Ex1_Grayscale import GrayscaleProcessor
from Week2_Filtering.Week2_Ex2_Gaussian import GaussianProcessor
from Week2_Filtering.Week2_Ex3_Median import MedianProcessor
from Week2_Filtering.Week2_Ex4_SobelX import SobelXProcessor
from Week2_Filtering.Week2_Ex5_Laplacian import LaplacianProcessor
from Week2_Filtering.Week2_Ex6_Sharpen import SharpenProcessor
from Week2_Filtering.Week2_Ex7_Bilateral import BilateralProcessor
from Week2_Filtering.Week2_Ex8_Threshold import ThresholdProcessor
from Week2_Filtering.Week2_Ex9_Erosion import ErosionProcessor
from Week2_Filtering.Week2_Ex10_Dilation import DilationProcessor


# CHANGE THIS for each experiment:
# 'grayscale', 'gaussian', 'median', 'sobelx', 'laplacian', 'sharpen',
# 'bilateral', 'threshold', 'erosion', 'dilation'
FILTER_MODE = "median"


class ImageProcessor:
    def __init__(self):
        pass

    def process_frame(self, bgr_img):
        if bgr_img is None:
            raise ValueError("Input frame is None")

        start_time = time.perf_counter()
        results = {}

        saver = CaptureSaveImgProcessor()
        saver.capture_and_save_image(bgr_img, "test_capture.bmp")

        processed = None

        if FILTER_MODE == "grayscale":
            processed = GrayscaleProcessor().convert_to_grayscale(bgr_img)

        elif FILTER_MODE == "gaussian":
            processed = GaussianProcessor().apply_gaussian_filter(bgr_img, kernel_size=(5, 5), sigma=1.0)

        elif FILTER_MODE == "median":
            processed = MedianProcessor().apply_median_blur(bgr_img, ksize=5)

        elif FILTER_MODE == "sobelx":
            processed = SobelXProcessor().sobel_x(bgr_img, ksize=3)

        elif FILTER_MODE == "laplacian":
            processed = LaplacianProcessor().laplacian(bgr_img, ksize=3)

        elif FILTER_MODE == "sharpen":
            processed = SharpenProcessor().sharpen(bgr_img)

        elif FILTER_MODE == "bilateral":
            processed = BilateralProcessor().bilateral(bgr_img, d=9, sigmaColor=75, sigmaSpace=75)

        elif FILTER_MODE == "threshold":
            processed = ThresholdProcessor().binary_threshold(bgr_img, thresh=127)

        elif FILTER_MODE == "erosion":
            processed = ErosionProcessor().erode(bgr_img, ksize=3, iterations=1)

        elif FILTER_MODE == "dilation":
            processed = DilationProcessor().dilate(bgr_img, ksize=3, iterations=1)

        else:
            processed = bgr_img  # fallback

        # safety: never return None
        if processed is None:
            processed = bgr_img

        saver.capture_and_save_image(processed, "processed_capture.bmp")

        process_time_ms = (time.perf_counter() - start_time) * 1000
        results["filter"] = FILTER_MODE

        return processed, results, process_time_ms
