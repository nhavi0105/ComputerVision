import cv2
import numpy as np

class LaplacianProcessor:
    def __init__(self):
        pass

    def laplacian(self, bgr_img, ksize=3):
        if bgr_img is None:
            return None
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        lap = np.uint8(np.clip(np.abs(lap), 0, 255))
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
