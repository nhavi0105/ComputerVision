import cv2

class ThresholdProcessor:
    def __init__(self):
        pass

    def binary_threshold(self, bgr_img, thresh=127):
        if bgr_img is None:
            return None
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
