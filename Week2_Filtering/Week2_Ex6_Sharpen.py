import cv2
import numpy as np

class SharpenProcessor:
    def __init__(self):
        pass

    def sharpen(self, bgr_img):
        if bgr_img is None:
            return None
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(bgr_img, -1, kernel)
