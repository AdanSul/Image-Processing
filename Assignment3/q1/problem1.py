import cv2
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error

# function to save images
def save_image(file_name, image):
    try:
        cv2.imwrite(file_name, image)
        print(f"Image saved successfully: {file_name}")
    except Exception as e:
        print(f"Error saving image {file_name}: {e}")

# Load all processed images at once
def load_processed_images():
    images = {}
    for i in range(1, 10): 
        images[f"image_{i}"] = cv2.imread(f"image_{i}.jpg", cv2.IMREAD_GRAYSCALE)
    return images

# Display images
def visualize_comparison(original, result, label1='Input', label2='Output'):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(label1)

    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.title(label2)

    plt.show()
    mse = mean_squared_error(original, result)
    print(f"MSE: {mse}")

# Image processing functions
def horizontal_blur_process(base_image, img):
    _, width = img.shape
    kernel = np.ones((1, 10000), np.float32) / 10000
    result = scipy.signal.convolve2d(base_image, kernel, mode='same', boundary='wrap')
    visualize_comparison(img, result)
    save_image('fixed_image_1.jpg', result)

def gaussian_blur_process(base_image, img):
    result = cv2.GaussianBlur(base_image, (13, 13), 13.0)
    visualize_comparison(img, result)
    save_image('fixed_image_2.jpg', result)

def median_filter_process(base_image, img):
    result = cv2.medianBlur(base_image, 11)  
    visualize_comparison(img, result)
    save_image('fixed_image_3.jpg', result)

def motion_blur_process(base_image, img):
    result = cv2.GaussianBlur(base_image, (1, 15), 17)
    visualize_comparison(img, result)
    save_image('fixed_image_4.jpg', result)

def bilateral_difference_process(base_image, img):
    filtered = cv2.bilateralFilter(base_image, 15, 240, 240)
    result = base_image - filtered + 127
    visualize_comparison(img, result)
    save_image('fixed_image_5.jpg', result)

def edge_detection_process(base_image, img):
    kernel = np.array([[-1 / 1900, -1, -1 / 1900],
                       [0, 0, 0],
                       [1 / 1900, 1, 1 / 1900]])
    result = cv2.filter2D(base_image, -1, kernel)
    visualize_comparison(img, result)
    save_image('fixed_image_6.jpg', result)

def convolution_process(base_image, img):
    height, width = img.shape
    kernel = np.zeros((height, width), np.float32)
    kernel[0, width // 2] = 1
    swapped_img = cv2.filter2D(base_image, -1, kernel, borderType=cv2.BORDER_WRAP)
    visualize_comparison(img, swapped_img)
    save_image('fixed_image_7.jpg', swapped_img)

def identity_filter_process(base_image, img):
    result = base_image
    visualize_comparison(img, result)
    save_image('fixed_image_8.jpg', result)

def weighted_blur_process(base_image, img):
    blurred = cv2.GaussianBlur(base_image, (9, 9), 0)
    result = cv2.addWeighted(base_image, 2.1, blurred, -1.1, 0)
    visualize_comparison(img, result)
    save_image('fixed_image_9.jpg', result)

if __name__ == "__main__":
    # Load base image and processed images
    base_image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    processed_images = load_processed_images()

    # Process each image
    horizontal_blur_process(base_image, processed_images['image_1'])
    gaussian_blur_process(base_image, processed_images['image_2'])
    median_filter_process(base_image, processed_images['image_3'])
    motion_blur_process(base_image, processed_images['image_4'])
    bilateral_difference_process(base_image, processed_images['image_5'])
    edge_detection_process(base_image, processed_images['image_6'])
    convolution_process(base_image, processed_images['image_7'])
    identity_filter_process(base_image, processed_images['image_8'])
    weighted_blur_process(base_image, processed_images['image_9'])
