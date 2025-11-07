import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, img_as_float

def parallel_cellular_edge_detection(image, threshold=0.2):
    """
    Perform edge detection using parallel cellular automata approach.
    
    Args:
        image (2D numpy array): Grayscale image with pixel values in [0,1].
        threshold (float): Difference threshold to detect edges.
        
    Returns:
        edges (2D numpy array): Binary edge map (1=edge, 0=non-edge).
    """
    # Pad image to handle border pixels
    padded = np.pad(image, pad_width=1, mode='edge')
    
    # Prepare empty edge map
    edges = np.zeros_like(image)
    
    # Define neighborhood relative coordinates (8 neighbors)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    
    # Vectorized computation of max difference with neighbors
    max_diff = np.zeros_like(image)
    
    for dy, dx in neighbors:
        neighbor_vals = padded[1 + dy : 1 + dy + image.shape[0], 
                               1 + dx : 1 + dx + image.shape[1]]
        diff = np.abs(image - neighbor_vals)
        max_diff = np.maximum(max_diff, diff)
    
    # Threshold difference to determine edges
    edges[max_diff > threshold] = 1
    
    return edges

# Load example image and convert to grayscale
image = color.rgb2gray(data.astronaut())
image = img_as_float(image)  # normalize to [0,1]

# Detect edges using cellular automata method
edges = parallel_cellular_edge_detection(image, threshold=0.2)

# Plot original and edge map
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Detected Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()
