import cv2
import numpy as np
import os

# ====== CONFIG ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ====== ROI FUNCTION ======
def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


# ====== DRAW LINES (FILTER SLOPE) ======
def draw_lines(img, lines):
    if lines is None:
        print("No lines detected")
        return img

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < 0.4:
            continue

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    if len(left_lines) == 0 and len(right_lines) == 0:
        print("No lane detected after filtering")
        return img

    def average_line(lines):
        if len(lines) == 0:
            return None
        slope_avg = np.mean([l[0] for l in lines])
        intercept_avg = np.mean([l[1] for l in lines])
        return slope_avg, intercept_avg

    left_avg = average_line(left_lines)
    right_avg = average_line(right_lines)

    line_img = np.zeros_like(img)

    height = img.shape[0]
    y1 = height
    y2 = int(height * 0.6)

    def make_points(line):
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    if left_avg is not None:
        x1, y1_, x2, y2_ = make_points(left_avg)
        cv2.line(line_img, (x1, y1_), (x2, y2_), (0, 255, 0), 6)

    if right_avg is not None:
        x1, y1_, x2, y2_ = make_points(right_avg)
        cv2.line(line_img, (x1, y1_), (x2, y2_), (0, 255, 0), 6)

    return cv2.addWeighted(img, 0.8, line_img, 1, 1)

    def average_line(lines):
        if len(lines) == 0:
            return None
        slope_avg = np.mean([l[0] for l in lines])
        intercept_avg = np.mean([l[1] for l in lines])
        return slope_avg, intercept_avg

    left_avg = average_line(left_lines)
    right_avg = average_line(right_lines)

    if left_avg is None and right_avg is None:
        print("No lane detected after filtering")
        return img

    line_img = np.zeros_like(img)

    height = img.shape[0]
    y1 = height
    y2 = int(height * 0.6)

    def make_points(line):
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    if left_avg is not None:
        x1, y1_, x2, y2_ = make_points(left_avg)
        cv2.line(line_img, (x1, y1_), (x2, y2_), (0, 255, 0), 6)

    if right_avg is not None:
        x1, y1_, x2, y2_ = make_points(right_avg)
        cv2.line(line_img, (x1, y1_), (x2, y2_), (0, 255, 0), 6)

    return cv2.addWeighted(img, 0.8, line_img, 1, 1)

def color_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])

    lower_yellow = np.array([10, 70, 70])
    upper_yellow = np.array([40, 255, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    return cv2.bitwise_or(white_mask, yellow_mask)

# ====== MAIN PROCESS ======
def process_image(path, output_name):
    print(f"\nProcessing: {path}")

    img = cv2.imread(path)

    if img is None:
        print("Cannot load image")
        return

    mask = color_filter(img)
    
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    
    kernel = np.ones((3,3), np.uint8)
    blur = cv2.dilate(blur, kernel, iterations=1)
    
    # canny
    median = np.median(blur)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    
    edges = cv2.Canny(blur, lower, upper)

    # ROI
    roi = region_of_interest(edges)

    # Hough Transform 
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=100,
    )

    # Draw filtered lines
    result = draw_lines(img, lines)

    # Save output
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    success = cv2.imwrite(output_path, result)

    if success:
        print(f"Saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")


# ====== LOOP THROUGH IMAGES ======
def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Folder not found: {INPUT_FOLDER}")
        return

    images = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(images) == 0:
        print("No images found")
        return

    print(f"Found {len(images)} images")

    for i, img_name in enumerate(images):
        path = os.path.join(INPUT_FOLDER, img_name)
        output_name = f"output_{i+1}.jpg"
        process_image(path, output_name)


if __name__ == "__main__":
    main()



