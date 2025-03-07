import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = image.shape

#------ section a ----------
fourier_transform = fft2(image)  # 2D Fourier Transform
shifted_fourier = fftshift(fourier_transform)  # Shift zero frequency to the center
magnitude_spectrum = np.log(1 + np.abs(shifted_fourier))  # Compute the magnitude spectrum


#------ section b ----------
def compute_fourier(image):
    return fftshift(fft2(image))

# Create a padded frequency array
def create_padded_frequency(fourier_data, original_shape):
    H, W = original_shape
    padded_H, padded_W = 2 * H, 2 * W
    padded_frequency = np.zeros((padded_H, padded_W), dtype=np.complex64)
    offset_H, offset_W = padded_H // 4, padded_W // 4
    padded_frequency[offset_H:offset_H + H, offset_W:offset_W + W] = fourier_data
    return padded_frequency

# SPerform inverse Fourier Transform and normalize
def scale_image_with_fourier(padded_frequency):
    scaled_image = np.fft.ifft2(ifftshift(padded_frequency)).real
    scaled_image *= 4  # Brightness correction
    return np.clip(scaled_image, 0, 255).astype(np.uint8)

original_fourier = compute_fourier(image)
padded_frequency = create_padded_frequency(original_fourier, image.shape)
scaled_image = scale_image_with_fourier(padded_frequency)

# Compute the spectrum for visualization
original_spectrum = np.log(1 + np.abs(original_fourier))
padded_spectrum = np.log(1 + np.abs(padded_frequency))


#------ section c ----------
def create_scaled_fourier_transform(fourier_data, scale_h, scale_w):
    new_transform = np.zeros((scale_h, scale_w), dtype=np.complex64)
    for u in range(scale_h):
        for v in range(scale_w):
            if u % 2 == 0 and v % 2 == 0:  # Only map even coordinates
                orig_u, orig_v = u // 2, v // 2
                if orig_u < h and orig_v < w:  # Check bounds
                    new_transform[u, v] = fourier_data[orig_u, orig_v] / 4
    return new_transform

# Generate scaled Fourier transform and image
scaled_fourier_transform = create_scaled_fourier_transform(shifted_fourier, 2 * h, 2 * w)
scaled_fourier_image = np.fft.ifft2(ifftshift(scaled_fourier_transform)).real
scaled_fourier_image = np.clip(scaled_fourier_image, 0, 255).astype(np.uint8)

# Compute log spectrum for visualization
scaled_fourier_spectrum = np.log(1 + np.abs(scaled_fourier_transform))


plt.figure(figsize=(10,10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(padded_spectrum, cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(scaled_image, cmap='gray')

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(scaled_fourier_spectrum, cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(scaled_fourier_image, cmap='gray')
plt.savefig('zebra_scaled.png')

plt.subplots_adjust(hspace=0.4)
plt.show()
