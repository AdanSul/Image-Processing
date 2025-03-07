import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def display_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histogram_equalization(image):
    fixed_image = cv2.equalizeHist(image)
    display_image("Histogram Equalization", fixed_image)
    return fixed_image

def adjust_contrast_brightness(image, alpha, beta):
    fixed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    display_image(f"Contrast {alpha}, Brightness {beta}", fixed_image)
    return fixed_image

def max_brightness_contrast_stretching(image):
    min_val, max_val, _, _ = cv2.minMaxLoc(image)
    alpha = 255.0 / (max_val - min_val) 
    beta = -alpha * min_val             
    fixed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    display_image(f"maximum brightness and contrast", fixed_image)
    return fixed_image

def gamma_correction(image, gamma):
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype="uint8")
    fixed_image = cv2.LUT(image, lookup_table)
    display_image(f"Gamma Correction (Gamma={gamma})", fixed_image)
    return fixed_image


def apply_fix(image, id):
    display_image("Original Image", image)

    if id == 1:
        print("Applying Histogram Equalization...")
        return histogram_equalization(image)
    elif id == 2:
        print("Applying Gamma Correction (0.65)...")
        return gamma_correction(image, gamma= 0.65)
    elif id == 3:
        print("Applying Maximum Contrast and Brightness...")
        return max_brightness_contrast_stretching(image)
    else:
        return image


def find_image_path(image_id):
    formats = ['.jpg', '.png']
    for fmt in formats:
        path = f"{image_id}{fmt}"
        if os.path.exists(path):
            return path
    return None

for i in range(1, 4):
    path = find_image_path(i)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Apply the fixes and get the chosen fixed image
    fixed_image = apply_fix(image, i)
    # Save the fixed image
    if i == 1:
        output_path = f"(OPTIONAL) {i} - Fixed.png"
    else:
        output_path = f"(OPTIONAL) {i} - Fixed.jpg"
    cv2.imwrite(output_path, fixed_image)
    print(f"Fixed Image {i} saved as {output_path}")

