import cv2
import numpy as np
import matplotlib.pyplot as plt

# Applies Gaussian blur and downsamples the image.
def downsample_image(image, scale_factor):
    blurred = cv2.GaussianBlur(image, (3, 3), 1)  # Apply Gaussian blur
    resized = cv2.resize(blurred, None, fx=scale_factor, fy=scale_factor)
    return resized

# Generates a Gaussian pyramid by progressively downsampling the image.
def get_gaussian_pyramid(image, levels, resize_ratio=0.5):
    gaussian_pyr = [image] 

    for _ in range(1, levels):
        image = downsample_image(image, resize_ratio)  
        gaussian_pyr.append(image)

    return gaussian_pyr

def get_laplacian_pyramid(image, levels, resize_ratio=0.5):
    image = np.float32(image.copy())
    # Generate the Gaussian pyramid
    gaussian_pyr = get_gaussian_pyramid(image, levels, resize_ratio)
    # Precompute upscaled images
    upscaled_images = {
        i: cv2.resize(gaussian_pyr[i + 1], (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        for i in range(levels - 1)
    }
    # Compute Laplacian pyramid 
    laplacian_pyr = [
        cv2.subtract(gaussian_pyr[i], upscaled_images[i]) for i in range(levels - 1)
    ]
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

# Expands an image using a given resize ratio.
def expand_image(image, resize_ratio):
    return cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)

def restore_from_pyramid(pyramidList, resize_ratio=2):
    restored_image = pyramidList[-1]  # Start with the smallest image
    
    for level, laplacian in enumerate(reversed(pyramidList[:-1])):
        restored_image = expand_image(restored_image, resize_ratio)  
        restored_image = cv2.add(restored_image, laplacian)  
    
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    
    return restored_image

def validate_operation(img):
	pyr = get_laplacian_pyramid(img, levels=5)
	img_restored = restore_from_pyramid(pyr)

	plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
	plt.imshow(img_restored, cmap='gray')

	plt.show()
	

# def blend_pyramids(levels):
#     total_levels = len(pyr_orange) 
#     blended_pyr = []

#     for curr_level in range(total_levels):
#         height, width = pyr_orange[curr_level].shape[:2]

#         # Create mask of same size as pyramid level
#         mask = np.zeros((height, width))

#         # Define transition points
#         mid_point = int(0.5 * width)
#         left_transition = max(0, mid_point - curr_level)
#         right_transition = min(width, mid_point + curr_level)

#         # Left side (fully orange)
#         mask[:, :left_transition] = 1.0

#         # Gradual transition (blending region) 
#         if curr_level > 0 and right_transition > left_transition:
#             mask[:, left_transition:right_transition] = 0.9 - (
#                 0.9 * (np.arange(right_transition - left_transition)) / (2 * curr_level)
#             )

#         # Apply Gaussian blur for smooth blending
#         mask = cv2.GaussianBlur(mask, (11, 11), 0)

#         # Blend using the mask
#         blended_level = pyr_orange[curr_level] * mask + pyr_apple[curr_level] * (1 - mask)
#         blended_pyr.append(blended_level)

#     return blended_pyr


# Enhanced blending function with a wider and smoother transition.
def blend_pyramids(levels):
    total_levels = len(pyr_orange)
    blended_pyr = []

    for curr_level in range(total_levels):
        height, width = pyr_orange[curr_level].shape[:2]

        # Create mask of same size as pyramid level
        mask = np.zeros((height, width), dtype=np.float32)

        transition_width = int(6 * (curr_level + 1))  # Wider transition
        if transition_width % 2 == 0:
            transition_width += 1  

        midpoint = width // 2
        left_transition = max(0, midpoint - transition_width)
        right_transition = min(width, midpoint + transition_width)

        # Left side (fully orange)
        mask[:, :left_transition] = 1.0

        # Smoother transition using np.linspace
        if right_transition > left_transition:
            mask[:, left_transition:right_transition] = np.linspace(1.0, 0.0, right_transition - left_transition)

        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        # Blend using the mask
        blended_level = pyr_orange[curr_level] * mask + pyr_apple[curr_level] * (1 - mask)
        blended_pyr.append(blended_level)

    return blended_pyr


apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

validate_operation(apple)
validate_operation(orange)

pyr_apple = get_laplacian_pyramid(apple, levels=6)
pyr_orange = get_laplacian_pyramid(orange, levels=6)

pyr_result = []

pyr_result = blend_pyramids(levels = 6)

final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()

cv2.imwrite("result.jpg", final)

