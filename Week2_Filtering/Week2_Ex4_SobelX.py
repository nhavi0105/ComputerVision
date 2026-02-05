import cv2
import numpy as np

class SobelXProcessor:
    def __init__(self):
        pass

    def sobel_x(self, bgr_img, ksize=3):
        if bgr_img is None:
            return None
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gx = np.uint8(np.clip(np.abs(gx), 0, 255))
        return cv2.cvtColor(gx, cv2.COLOR_GRAY2BGR)
