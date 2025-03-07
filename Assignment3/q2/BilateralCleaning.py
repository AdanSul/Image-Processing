import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load and convert images to grayscale
images = {
    "taj": cv2.imread("taj.jpg", cv2.IMREAD_GRAYSCALE),
    "NoisyGrayImage": cv2.imread("NoisyGrayImage.png", cv2.IMREAD_GRAYSCALE),
    "balls": cv2.imread("balls.jpg", cv2.IMREAD_GRAYSCALE)
}
# Parameters for each image 
params = {
    "taj": {"radius": 7, "stdSpatial": 80, "stdIntensity": 30},
    "NoisyGrayImage": {"radius": 7, "stdSpatial": 25, "stdIntensity": 100},
    "balls": {"radius": 6, "stdSpatial": 10, "stdIntensity": 9}
}

def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):

    im = im.astype(np.float64)
    rows, cols = im.shape

    cleanIm = np.zeros_like(im)

    # Precompute the spatial weight (gs)
    offset = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(offset, offset)
    gs = np.exp(-(x**2 + y**2) / (2 * stdSpatial**2))

    # Iterate over each pixel 
    for i in range(rows):
        for j in range(cols):
            # Define the local window bounds
            r_min = max(i - radius, 0)
            r_max = min(i + radius + 1, rows)
            c_min = max(j - radius, 0)
            c_max = min(j + radius + 1, cols)

            window = im[r_min:r_max, c_min:c_max]

            # Calculate intensity weight
            gi = np.exp(-((window - im[i, j])**2) / (2 * stdIntensity**2))

            gs_window = gs[(r_min - i + radius):(r_max - i + radius),
                           (c_min - j + radius):(c_max - j + radius)]

            combined_weights = gi * gs_window

            # Normalize weights
            combined_weights /= combined_weights.sum()

            # Compute the new pixel value
            cleanIm[i, j] = np.sum(combined_weights * window)

    # Clip and convert back to uint8
    cleanIm = np.clip(cleanIm, 0, 255).astype(np.uint8)

    return cleanIm


# Apply bilateral filter to each image and save results
filtered_images = {}
for name, img in images.items():
    p = params[name]
    filtered_images[name] = clean_Gaussian_noise_bilateral(
        img, radius=p["radius"], stdSpatial=p["stdSpatial"], stdIntensity=p["stdIntensity"]
    )
    cv2.imwrite(f"cleaned_{name}.jpg", filtered_images[name])

# Display images
for name, img in images.items():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original {name}")
    plt.axis('off')
    plt.imshow(img, cmap="gray")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Cleaned {name}")
    plt.axis('off')
    plt.imshow(filtered_images[name], cmap="gray")
    
    plt.show()
