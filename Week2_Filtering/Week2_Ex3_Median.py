import cv2

class MedianProcessor:
    def __init__(self):
        pass

    def apply_median_blur(self, img, ksize=45):
        if img is None:
            return None

        # force odd ksize
        if ksize % 2 == 0:
            ksize += 1
        if ksize < 3:
            ksize = 3

        return cv2.medianBlur(img, ksize)
