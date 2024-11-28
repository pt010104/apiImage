import cv2
import numpy as np
import os

def handle_cropped_image(cropped_image):
    gray_image = cropped_image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned_image = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    closed_kernal = np.ones((7, 7), np.uint8)
    open_kernal = np.ones((1, 2), np.uint8)
    opened_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, open_kernal)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, closed_kernal)
    (cnt, hierarchy) = cv2.findContours(closed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(cnt, key=cv2.contourArea, reverse=True)
    mask = np.zeros_like(gray_image)
    for c in sorted_contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), -1)
    binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    masked_image = cv2.bitwise_and(gray_image, gray_image, mask=binary_mask)
    masked_image = cv2.bitwise_not(masked_image)
    return masked_image

def process_image(image_path):
    original_image = cv2.imread(image_path)
    scale = 1 / 3
    width = int(original_image.shape[1] * scale) 
    height = int(original_image.shape[0] * scale) 
    dim = (width, height)
    original_image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    average_intensity = np.mean(gray_image)
    if average_intensity > 127:
        gray_image = cv2.bitwise_not(gray_image)
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=3)
    line_positions = []
    line_mask = np.zeros_like(edges)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 3:
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                line_positions.append((x1, 0, 0, original_image.shape[0]))
    line_positions = sorted(line_positions, key=lambda pos: pos[0])
    line_mask = cv2.bitwise_not(line_mask)
    gray_image = cv2.bitwise_and(gray_image, gray_image, mask=line_mask)
    if line_positions:
        line_positions = [(0, 0, 0, original_image.shape[0])] + line_positions
        line_positions.append((original_image.shape[1], 0, 0, original_image.shape[0]))

        cropped_images = []
        for i in range(len(line_positions) - 1):
            x1 = line_positions[i][0]
            x2 = line_positions[i + 1][0]

            if x1 < x2:
                cropped = gray_image[:, x1:x2]
                cropped_images.append(cropped)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cropped_images = [gray_image]
    for i, cropped_image in enumerate(cropped_images):
        cropped_images[i] = handle_cropped_image(cropped_image)

    final_image = np.hstack(cropped_images)
    return final_image

output_images_dir = "processed_images"
os.makedirs(output_images_dir, exist_ok=True)
folder_path = "extracted_images"
for filename in os.listdir(folder_path):
    if filename.endswith(".jpeg"):
        image_path = os.path.join(folder_path, filename)
        final_image = process_image(image_path)
        output_image_path = os.path.join(output_images_dir, filename)
        cv2.imwrite(output_image_path, final_image)
