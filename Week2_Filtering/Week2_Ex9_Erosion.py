import cv2
import numpy as np

class ErosionProcessor:
    def __init__(self):
        pass

    def erode(self, bgr_img, ksize=3, iterations=1):
        if bgr_img is None:
            return None
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.erode(bgr_img, kernel, iterations=iterations)
