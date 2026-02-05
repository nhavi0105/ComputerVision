import cv2

class BilateralProcessor:
    def __init__(self):
        pass

    def bilateral(self, bgr_img, d=9, sigmaColor=75, sigmaSpace=75):
        if bgr_img is None:
            return None
        return cv2.bilateralFilter(bgr_img, d, sigmaColor, sigmaSpace)
