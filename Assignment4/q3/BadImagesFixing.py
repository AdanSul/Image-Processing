import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


def get_transformation_points():
    return [
        (
            np.float32([[6, 20], [111, 20], [6, 130], [111, 130]]),  # Source for image 1
            np.float32([[0, 0], [255, 0], [0, 255], [255, 255]])   # Destination
        ),
        (
            np.float32([[78, 162], [146, 117], [135, 245], [244, 160]]),  # Source for image 2
            np.float32([[0, 0], [255, 0], [0, 255], [255, 255]])         # Destination
        ),
        (
            np.float32([[180, 5], [250, 70], [180, 120], [122, 52]]),  # Source for image 3
            np.float32([[0, 0], [255, 0], [255, 255], [0, 255]])       # Destination
        )
    ]

def clean_baby(im):
    denoised_im = cv2.medianBlur(im, ksize=3)
    # Perform perspective transformations
    transformed_images = []
    points = get_transformation_points()
    for source, destination in points:
        transformation_matrix = cv2.getPerspectiveTransform(source, destination)
        transformed_image = cv2.warpPerspective(
            denoised_im, transformation_matrix, (256, 256), flags=cv2.INTER_CUBIC
        )
        transformed_images.append(transformed_image)

    # Combine the transformed images using a median filter
    combined_image = np.median(np.array(transformed_images), axis=0)
    return combined_image.astype(np.uint8)


def clean_windmill(im):
    freq_domain = np.fft.fft2(im)  
    shifted_freq = np.fft.fftshift(freq_domain)  

    # These offsets represent the distance of the noise frequencies from the center
    noise_offsets = [(4, 28), (-4, -28)]

    # the center of the frequency spectrum
    center_row, center_col = im.shape[0] // 2, im.shape[1] // 2

    # Suppress noise frequencies based on the calculated offsets
    for offset in noise_offsets:
        row_offset, col_offset = offset
        shifted_freq[center_row + row_offset, center_col + col_offset] = 0  # Suppress noise
        shifted_freq[center_row - row_offset, center_col - col_offset] = 0  # Suppress symmetric noise

    # Transform back to the spatial domain
    unshifted_freq = np.fft.ifftshift(shifted_freq)  # Shift frequencies back
    cleaned_image = np.fft.ifft2(unshifted_freq)  # Inverse Fourier Transform

    return np.abs(cleaned_image)


def clean_watermelon(im):
    sharpening_kernel = np.array([[0, -4,  0],[-4,  17, -4],[0, -4,  0]])
    sharpened_image = cv2.filter2D(im, -1, sharpening_kernel)

    return sharpened_image

import numpy as np

def clean_umbrella(im):
    freq_domain = np.fft.fft2(im)
    # Build the mask 
    def generate_mask(shape):
        mask = np.zeros(shape)
        mask[4, 79] = 0.5  
        mask[0, 0] = 0.5    
        return mask

    mask = generate_mask(im.shape)
    mask_freq = np.fft.fft2(mask)
    # avoid division errors
    safe_mask = np.where(abs(mask_freq) < 0.01, 1, mask_freq)
    # Filter the image in the frequency domain
    filtered_freq = freq_domain / safe_mask
    cleaned_image = np.abs(np.fft.ifft2(filtered_freq))
    cleaned_image = (255 * (cleaned_image - cleaned_image.min()) /
                     (cleaned_image.max() - cleaned_image.min()))

    return cleaned_image.astype(np.uint8)



def clean_USAflag(im):
    clean_im = im.copy()
    # Apply a median filter and blur to the lower portion
    clean_im[90:, :] = median_filter(im[90:, :], size=(1, 50))  
    clean_im[90:, :] = cv2.blur(clean_im[90:, :], ksize=(15, 1))
    # Apply a median filter and blur to the upper-right portion
    clean_im[:90, 142:] = median_filter(im[:90, 142:], size=(1, 50))
    clean_im[:90, 142:] = cv2.blur(clean_im[:90, 142:], ksize=(15, 1))

    return clean_im

def clean_house(im):
    fft_image = np.fft.fft2(im)

    def create_mask(rows, cols, threshold=0.02):
        base_mask = np.zeros((rows, cols))
        base_mask[0, :10] = 0.1
        mask_fft = np.fft.fft2(base_mask)
        return np.where(abs(mask_fft) < threshold, 1, mask_fft)

    mask = create_mask(*im.shape)

    # Correct the Fourier-transformed image using the mask
    corrected_fft_image = fft_image / mask

    # Inverse Fourier Transform 
    cleaned_image = np.fft.ifft2(corrected_fft_image)
    cleaned_image = np.abs(cleaned_image)

    # Normalize the result to the 0-255 range
    cleaned_image = (255 * (cleaned_image - cleaned_image.min()) / 
                     (cleaned_image.max() - cleaned_image.min()))

    return cleaned_image.astype(np.uint8)


def clean_bears(im):
    # Normalize the image
    normalized_im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Apply CLAHE for adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_im = clahe.apply(normalized_im)
    
    return enhanced_im


