import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy.lib.stride_tricks import sliding_window_view

import warnings
warnings.filterwarnings("ignore")


def crop_frequency_domain(f_transform, new_h, new_w):
    h, w = f_transform.shape
    center_h, center_w = h // 2, w // 2
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    return f_transform[
        center_h - new_h // 2 : center_h + new_h // 2,
        center_w - new_w // 2 : center_w + new_w // 2
    ]

def inverse_fourier_transform(f_transform):
    f_transform = ifftshift(f_transform)
    scaled_image = np.abs(ifft2(f_transform))

    scaled_image = cv2.normalize(scaled_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return scaled_image.astype(np.uint8)

def scale_down(image, resize_ratio):
    image = image.astype(np.float32)
    f_transform = fftshift(fft2(image))

    h, w = image.shape
    new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)

    cropped_f_transform = crop_frequency_domain(f_transform, new_h, new_w)

    return inverse_fourier_transform(cropped_f_transform)

def pad_frequency_domain(f_transform, new_h, new_w):
    h, w = f_transform.shape
    pad_h, pad_w = (new_h - h) // 2, (new_w - w) // 2 
    padded_f_transform = np.zeros((new_h, new_w), dtype=complex)
    padded_f_transform[pad_h:pad_h + h, pad_w:pad_w + w] = f_transform
    return padded_f_transform

def scale_up(image, resize_ratio):
    image = image.astype(np.float32)

    f_transform = np.fft.fftshift(np.fft.fft2(image))
    h, w = image.shape
    new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)

    padded_f_transform = pad_frequency_domain(f_transform, new_h, new_w)

    shifted_f = np.fft.ifftshift(padded_f_transform)
    resized_image = np.abs(np.fft.ifft2(shifted_f))
    
    return np.uint8(cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX))

def ncc_2d(image, pattern):
    pattern_h, pattern_w = pattern.shape
    # Compute pattern mean and normalize it
    pattern_mean = np.mean(pattern)
    pattern_centered = pattern - pattern_mean
    pattern_norm = np.linalg.norm(pattern_centered)
    image_windows = sliding_window_view(image, (pattern_h, pattern_w))

    # Compute mean and normalize image windows
    windows_mean = np.mean(image_windows, axis=(2, 3), keepdims=True)
    windows_centered = image_windows - windows_mean
    windows_norm = np.linalg.norm(windows_centered, axis=(2, 3))

    # Compute NCC 
    denominator = pattern_norm * windows_norm
    denominator[denominator == 0] = 1  # Prevent division by zero
    ncc_map = np.sum(windows_centered * pattern_centered, axis=(2, 3)) / denominator

    return ncc_map

def display(image, pattern):
	
	plt.subplot(2, 3, 1)
	plt.title('Image')
	plt.imshow(image, cmap='gray')
		
	plt.subplot(2, 3, 3)
	plt.title('Pattern')
	plt.imshow(pattern, cmap='gray', aspect='equal')
	
	ncc = ncc_2d(image, pattern)
	
	plt.subplot(2, 3, 5)
	plt.title('Normalized Cross-Correlation Heatmap')
	plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto') 
	
	cbar = plt.colorbar()
	cbar.set_label('NCC Values')
		
	plt.show()

def draw_matches(image, matches, pattern_size):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	for point in matches:
		y, x = point
		top_left = (int(x - pattern_size[1]/2), int(y - pattern_size[0]/2))
		bottom_right = (int(x + pattern_size[1]/2), int(y + pattern_size[0]/2))
		cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)
	
	plt.imshow(image, cmap='gray')
	plt.show()
	
	cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)



CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
# display(image, pattern)

############# Students #############
image_resize_ratio = 1.32
pattern_resize_ratio = 0.72 
threshold =  0.488

image_scaled = scale_up(image, image_resize_ratio)
pattern_scaled = scale_down(pattern, pattern_resize_ratio)

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = np.argwhere(ncc > threshold)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.
real_matches = (real_matches / image_resize_ratio).astype(int)

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"

############# Crew #############
CURR_IMAGE = "thecrew"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_resize_ratio = 1.70
pattern_resize_ratio = 0.4
threshold = 0.45

image_scaled = scale_up(image, image_resize_ratio)
pattern_scaled =  scale_down(pattern, pattern_resize_ratio)

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = np.argwhere(ncc > threshold)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.
real_matches = (real_matches / image_resize_ratio).astype(int)

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"


