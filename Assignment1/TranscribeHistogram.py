import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
	img_size = imgs_arr[0].shape
	res = []
	
	for img in imgs_arr:
		X = img.reshape(img_size[0] * img_size[1], 1)
		km = KMeans(n_clusters=n_colors)
		km.fit(X)
		
		img_compressed = km.cluster_centers_[km.labels_]
		img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

		res.append(img_compressed.reshape(img_size[0], img_size[1]))
	
	return np.array(res)

# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
	image_arrays = []
	lst = [file for file in os.listdir(folder) if file.endswith(formats)]
	for filename in lst:
		file_path = os.path.join(folder, filename)
		image = cv2.imread(file_path)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image_arrays.append(gray_image)
	return np.array(image_arrays), lst

# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
	# Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
	x_pos = 70 + 40 * idx
	y_pos = 274
	while image[y_pos, x_pos] == 1:
		y_pos-=1
	return 274 - y_pos

# Sections c, d
def compare_hist(src_image, target):
    emd_threshold = 260  

    digit_height = target.shape[0]
    digit_width = target.shape[1]

    target_histogram = cv2.calcHist([target.astype(np.uint8)], channels=[0], mask=None, histSize=[256], ranges=[0, 256]).flatten()

    sliding_windows = np.lib.stride_tricks.sliding_window_view(src_image, (digit_height, digit_width))

    # Calculate the cumulative histogram of the target
    cumulative_target_histogram = np.cumsum(target_histogram)

    # Iterate through each window in the source image
    for row in range(sliding_windows.shape[0]):
        for col in range(sliding_windows.shape[1]):
            # Extract the current window
            current_window = sliding_windows[row, col]

            # Calculate the histogram of the current window
            window_histogram = cv2.calcHist([current_window], [0], None, [256], [0, 256]).flatten()

            # Calculate Earth Mover's Distance (EMD)
            cumulative_window_histogram = np.cumsum(window_histogram)
            absolute_difference = np.abs(cumulative_window_histogram - cumulative_target_histogram)
            emd_distance = np.sum(absolute_difference)

            # Check if the EMD distance is below the threshold
            if emd_distance < emd_threshold:
                return True # Region found

    return False # No region found

def crop_regions_of_interest(images, crop_params):
    cropped_regions = []  # Store the cropped regions

    # Extract crop parameters
    start_x, start_y, crop_width, crop_height = crop_params

    # Loop through all images and crop the region of interest
    for img_index, image in enumerate(images):
        # Crop the defined region
        cropped_region = image[
            start_y:start_y + crop_height, start_x:start_x + crop_width
        ]
        cropped_regions.append(cropped_region)

    # Convert the cropped regions list into a numpy array
    cropped_array = np.array(cropped_regions)
    return cropped_array


def calculate_highest_bars(images):
    highest_bars = [] 

    for img_idx, quantized_img in enumerate(images):  # Iterate through each image
        # Create a binary mask for the image (threshold set to 220)
        binary_mask = np.zeros(quantized_img.shape)
        binary_mask[quantized_img <= 220] = 1

        max_bar_height = 0  
        # Iterate through each bin in the histogram 
        for bin_idx in range(10):
            bar_height = get_bar_height(binary_mask, bin_idx)
            if bar_height > max_bar_height: 
                max_bar_height = bar_height

        highest_bars.append(max_bar_height)  

    return highest_bars

	
# Sections a, b
images, names = read_dir('data')
numbers, number_names  = read_dir('numbers')

# cv2.imshow(names[0], images[0]) 
# cv2.imshow(number_names[0],numbers[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

# # ---------- section b ----------
# highest_bars = calculate_highest_bars(images)

# # Print the results for verification
# for img_idx, bar_height in enumerate(highest_bars):
#     print(f"Histogram {names[img_idx]} highest bar in pixels along the vertical axis is {bar_height}")
# # ---------------------------------

# Crop parameters
crop_params = (20, 70, 25, 60)  

# Crop the regions of interest from the source images
cropped_images = crop_regions_of_interest(images, crop_params)
 

## -------- section c  --------
# plt.figure(figsize=(10, 10))
# for idx, cropped_image in enumerate(cropped_images):
#     plt.subplot(1, len(cropped_images), idx + 1)
#     plt.imshow(cropped_image, cmap='gray')
#     plt.title(f"{names[idx]}")
#     plt.axis('off')
# plt.show()

# digit_found = compare_hist(cropped_images[0],  numbers[6])

# if digit_found:
#     print(f"The digit in '{number_names[6]}' was found in the source image '{names[0]}'.")
# else:
#     print(f"The digit in '{number_names[6]}' was NOT found in the source image '{names[0]}'.")
# # ---------------------------------------------------------


#--------------- section d -------------------
max_students_per_histogram = []
recognized_digit = None 

for img_idx, image in enumerate(cropped_images):
	for digit_idx in range(9, -1, -1):
		target_image = numbers[digit_idx]  

		digit_found = compare_hist(image, target_image)

		if digit_found:
			recognized_digit = digit_idx
			# print(f"Image '{names[img_idx]}' was recognized as digit: {recognized_digit}")
			break 
    
	max_students_per_histogram.append(recognized_digit)
## ----------------------------------------


#--------------- section e -------------------
# Quantize the first image with different gray levels
gray_levels = [1, 2, 3, 4] 
quantized_images = []

first_image = images[0] 

# # Apply quantization for each gray level
# for n_colors in gray_levels:
#     quantized_image = quantization([first_image], n_colors=n_colors)[0] 
#     quantized_images.append((n_colors, quantized_image)) 
#     # Display the quantized image
#     cv2.imshow(f"Quantized Image with {n_colors} Gray Levels", quantized_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# Quantize all images with the optimal number of gray levels
optimal_gray_levels = 3  
quantized_images = quantization(images, n_colors=optimal_gray_levels)
binary_images = [] 

for idx, quantized_image in enumerate(quantized_images):
    binary_mask = np.zeros(quantized_image.shape, dtype=np.uint8)
    binary_mask[quantized_image <= 220] = 1
    
    binary_images.append(binary_mask)
    
    # cv2.imshow(f"Binary Image - {names[idx]}", binary_mask * 255)  # Multiply by 255 for proper visualization
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#--------------- section f ------------------

all_bar_heights = []  

for img_idx, binary_image in enumerate(binary_images):  # Loop through each binary image
    bar_heights = []  

    for bin_id in range(10):
        height = get_bar_height(binary_image, bin_id)  
        bar_heights.append(height)  

    all_bar_heights.append(bar_heights)  

    # print(f"Bar heights for histogram '{names[img_idx]}': {bar_heights}")

# ------------ section g -------------
for id, bar_heights in enumerate(all_bar_heights):  
    
    max_students = max_students_per_histogram[id]  
    max_bin_height = max(bar_heights) 

    heights = [
        round(max_students * bin_height / max_bin_height) for bin_height in bar_heights
    ]
    
    print(f'Histogram {names[id]} gave {heights}')
    
exit()