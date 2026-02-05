import cv2
import numpy as np

class DilationProcessor:
    def __init__(self):
        pass

    def dilate(self, bgr_img, ksize=3, iterations=1):
        if bgr_img is None:
            return None
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.dilate(bgr_img, kernel, iterations=iterations)
