import cv2  
import numpy as np  
from matplotlib import pyplot as plt  


# display and save the image
def display_and_save_image(image, title, filename):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    if not cv2.imwrite(filename, image):
        raise IOError(f"Failed to save the file '{filename}'.")

# Section A:
def fix_image_using_bilateral_and_median_filter(img_path, output_path, diameter, sigma_color, sigma_space, median_ksize):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply median filter
    median_result = cv2.medianBlur(img, median_ksize)

    # Apply bilateral filter
    bilateral_result = cv2.bilateralFilter(median_result, diameter, sigma_color, sigma_space)

    # Display and save the result
    display_and_save_image(bilateral_result, "Fixed Image - Section A", output_path)
    

# Section B: 
def fix_image_using_mean(npy_file, output_path):
    # Load the noisy images
    noised_images = np.load(npy_file)
   
    fixed_img = np.mean(noised_images, axis=0)
    
    # Normalize the image 
    fixed_img = cv2.normalize(fixed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Display and save the result
    display_and_save_image(fixed_img, "Fixed Image - Section B", output_path)


# Main Execution
if __name__ == "__main__":
    # Paths
    broken_img_path = 'broken.jpg'
    section_a_output_path = 'sectionA_fixed.jpg'
    npy_file_path = 'noised_images.npy'
    section_b_output_path = 'sectionB_fixed.jpg'

    # Section A: 
    print("Processing Section A...")
    fix_image_using_bilateral_and_median_filter(
        img_path=broken_img_path,
        output_path=section_a_output_path,
        diameter=9,
        sigma_color=80,
        sigma_space=80,
        median_ksize=5
    )
    
    # Section B:
    print("Processing Section B...")
    fix_image_using_mean(
        npy_file=npy_file_path,
        output_path=section_b_output_path
    )
