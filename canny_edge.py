import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def update(value):
    thresholded_magnitude = gradient_magnitude.copy()
    thresholded_magnitude, final_edges = canny_edge_detection(image, gradient_magnitude, gradient_direction, 150)
    
    ax4.clear()
    ax4.imshow(thresholded_magnitude, cmap='gray')
    ax4.set_title('Thresholded Gradient Magnitude')
    ax4.axis('off')
    
    ax5.clear()
    ax5.imshow(final_edges, cmap='gray')
    ax5.set_title('Traced Edges')
    ax5.axis('off')
    plt.draw()

def canny_edge_detection(image,gradient_magnitude,gradient_direction,value):
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    # Convert gradient direction to degrees
    gradient_direction = np.degrees(gradient_direction)
    
    # Quantize gradient direction into 4 directions: 0, 45, 90, 135 degrees
    quantized_direction = np.round(gradient_direction / 45) % 4
    
    # Apply non-maximum suppression
    suppressed = np.zeros_like(gradient_magnitude)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            if quantized_direction[i, j] == 0:
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif quantized_direction[i, j] == 1:
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
            elif quantized_direction[i, j] == 2:
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            else:
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
            
            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]
    
    # Thresholding and edge tracking by hysteresis
    strong_pixel = 255
    weak_pixel = 50
    high_threshold = value
    low_threshold = 30
    strong_edges = suppressed > high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    final_edges = np.zeros_like(suppressed)
    f2 = np.zeros_like(suppressed)
    f2[strong_edges] = 255
    f2[weak_edges] = 50
    final_edges[strong_edges] = 255
    final_edges[weak_edges] = 50
    

    rows, cols = final_edges.shape
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            if final_edges[row,col] == 50 or final_edges[row,col]==255:
                # Check if any neighboring pixel is a strong edge
                print(1)
                if np.any(final_edges[row-2:row+3, col-2:col+3] == strong_pixel):
                    final_edges[row, col] = strong_pixel
                else:
                    final_edges[row, col] = 0
    dfs(final_edges)

    return final_edges,f2

def trace_edges(rows, cols,row,col,final_edges):
    if 0 <= row < rows and 0 <= col < cols and final_edges[row, col] == 50:
        final_edges[row, col] = 255
        for r in range(row - 2, row + 3):
            for c in range(col - 2, col + 3):
                if (r != row or c != col) and 0 <= r < rows and 0 <= c < cols:
                    print(1)
                    trace_edges(final_edges.shape[0],final_edges.shape[1],r,c,final_edges)


def dfs(final_edges):
    for row in range(final_edges.shape[0]):
        for col in range(final_edges.shape[1]):
            if final_edges[row, col] == 255:
                print(1)
                trace_edges(final_edges.shape[0],final_edges.shape[1],row,col,final_edges)



# Load the image
image_path = "D://PythonApplication1//1234.jpg"  # Replace with the actual path of your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
edges = cv2.Canny(image, 30, 150)

# Display the original image and the edge-detected image
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
blurred = cv2.GaussianBlur(image,(3,3),1.7 )

# Calculate gradients
gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Calculate gradient magnitude and direction
gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
gradient_direction = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)

# Create a figure and subplots
fig, ((ax1, ax2), (ax3, ax4), (ax6, ax5)) = plt.subplots(3, 2, figsize=(10, 12))

ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(gradient_x, cmap='gray')
ax2.set_title('Gradient in X Direction')
ax2.axis('off')

ax3.imshow(gradient_y, cmap='gray')
ax3.set_title('Gradient in Y Direction')
ax3.axis('off')

ax4.imshow(gradient_magnitude, cmap='gray')
ax4.set_title('Gradient Magnitude')
ax4.axis('off')

# Add slider
ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03])
slider = Slider(ax_slider, 'Threshold', valmin=0, valmax=np.max(gradient_magnitude), valinit=0)
update(100)
plt.tight_layout()
plt.show()