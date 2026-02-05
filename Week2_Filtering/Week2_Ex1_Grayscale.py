import cv2


class GrayscaleProcessor:
    def __init__(self):
        pass
    
    # =============================================================================
    # STEP 1: BASIC IMAGE CAPTURE (Weeks 1-2)
    # Topic: Introduction to Computer Vision, Images as Functions & Filtering
    # =============================================================================

    def convert_to_grayscale(self, bgr_img):
        if bgr_img is None:
            return None

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        # Convert back to BGR so the rest of pipeline/UI stays consistent (3 channels)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray_bgr

