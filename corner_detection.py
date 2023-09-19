
from math import pi
from typing import Self
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.linalg import det


class edge:
    def update(value):
        thresholded_magnitude = gradient_magnitude.copy()
        thresholded_magnitude, final_edges = canny_edge_detection(image, gradient_magnitude,gradient_direction, 150)
    
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

    def canny():
        image_path = "d://computer vision//1234.jpg" # replace with the actual path of your image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # convert to grayscale
        edges = cv2.canny(image, 30, 150)

        # # display the original image and the edge-detected image
        cv2.imshow('original image', image)
        cv2.imshow('edges', edges)
        blurred = cv2.gaussianblur(image,(3,3),1.7 )

        # calculate gradients
        gradient_x = cv2.sobel(blurred, cv2.cv_64f, 1, 0, ksize=3)
        gradient_y = cv2.sobel(blurred, cv2.cv_64f, 0, 1, ksize=3)

        # calculate gradient magnitude and direction
        gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
        gradient_direction = cv2.phase(gradient_x, gradient_y, angleindegrees=true)

        # create a figure and subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax6, ax5)) = plt.subplots(3, 2, figsize=(10, 12))

        ax1.imshow(image, cmap='gray')
        ax1.set_title('original image')
        ax1.axis('off')

        ax2.imshow(gradient_x, cmap='gray')
        ax2.set_title('gradient in x direction')
        ax2.axis('off')

        ax3.imshow(gradient_y, cmap='gray')
        ax3.set_title('gradient in y direction')
        ax3.axis('off')

        ax4.imshow(gradient_magnitude, cmap='gray')
        ax4.set_title('gradient magnitude')
        ax4.axis('off')

        # add slider
        ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03])
        slider = slider(ax_slider, 'threshold', valmin=0, valmax=np.max(gradient_magnitude), valinit=0)
        update(100)
        plt.tight_layout()
        plt.show()

    def dfs(final_edges):
        for row in range(final_edges.shape[0]):
            for col in range(final_edges.shape[1]):
                if final_edges[row, col] == 255:
                    print(1)
                    trace_edges(final_edges.shape[0],final_edges.shape[1],row,col,final_edges)

    def furior_transform(y,x,image):
        _F = 0
        N,M = image.shape
        startu = int(-(image.shape[0]/2))
        startv = int(-(image.shape[1]/2))
        xcount = -1
        for u in range(startu,image.shape[0]+startu-1):
            xcount += 1
            ycount = -1
            for v in range(startv,image.shape[1]+startv-1):
                ycount+=1
                term = x*u/float(M) + y*v/float(N)
                _F += image[xcount,ycount]*np.exp(-2*np.pi*1j*term)
                #print(_F)
        _F = _F/np.prod(image.shape)
        return(_F)

    def plot_nonzero_Fourier_coefficient_line(_y,_x,_N,_M):
        ## horizontal and vertical lines along nonzero Fourier coefficient
        if _x < _M/2: ## x is always positive
            plt.axvline(_x)
            plt.axvline(-_x)
        else: ## x is not between - M/2,..., M/2-1
            plt.axvline(_M - _x)
            plt.axvline(-_M + _x)
        if _y < _N/2:
            plt.axhline(_y)
            plt.axhline(-_y)
        else: ## y is not between - N/2,..., N/2-1
            plt.axhline(_N - _y)
            plt.axhline(-_N + _y)

    def test(image):
        orig_img = image
        _N,_M = image.shape
        ySample, xSample = _N, _M
        _x,_y = _N,_M
        fs = np.zeros((ySample,xSample),dtype=complex)

        startx = int(-orig_img.shape[1]/2) 
        starty = int(-orig_img.shape[0]/2)

        countx = -1 
        xs = range(startx,startx + orig_img.shape[1])
        ys = range(starty,starty + orig_img.shape[0])
        for x in xs:
            countx += 1
            county = -1
            for y in ys:
                county += 1
                fs[countx,county] = furior_transform(y,x,orig_img)
                if np.abs(fs[countx,county]) > 0.0001:
                    print("x={:2.0f}, y={:2.0f}, f={:+5.3f}".format(x,y,fs[county,countx]))
        ploat(fs)

    def ploat(fs):   
        magnitude_fs = np.abs(fs)  # Compute the magnitude of complex values
        plt.imshow(magnitude_fs, cmap="gray")
        plt.colorbar(label="Magnitude")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Fourier Transform Magnitude")
        plt.show()

    def bfft(image):
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        ploat(fft_shift)

    def bifft(image,frquency):
        fillter = np.ones_like(image)
        fillter[:frquency] = 0
        image = image*fillter
        r_image = np.fft.ifft2(image)
        r_image = np.fft.ifftshift(r_image)
        ploat(r_image)
    
    def furior(angle,magnitude):
        width, height = 500,500
        image_real = np.zeros((height, width), dtype=np.float32)
        image_imag = np.zeros((height, width), dtype=np.float32)
    
        # Center of the image
        center_x, center_y = width // 2, height // 2
        angle_deg=angle
        freq_factor = magnitude
        angle_rad = np.radians(angle_deg)
        k_x = freq_factor * np.sin(angle_rad)
        k_y = freq_factor * np.cos(angle_rad)
        # Loop through frequencies and orientations
        # Try different frequency values
        # Generate sinusoidal pattern
        for y in range(height):
            for x in range(width):
                print("furior")
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                value_real = np.sin(k_x * x + k_y * y)
                value_imag = np.cos(k_x * x + k_y * y)
                image_real[y, x] += value_real
                image_imag[y, x] += value_imag
        # Display the image
        plt.imshow(image_real, cmap='gray')
        plt.axis('off')
        plt.show()
        return image_real
    
class ThreeVison:
    def normalStero_matching(img_left, img_right, max_disparity):
        height, width = img_left.shape
        disparity_map = np.zeros((height, width), np.uint8)
        cost_matrix = np.zeros((width, width), np.float32)
        direction_matrix = np.zeros((width, width), np.uint8)

        for y in range(height):
            cost_matrix.fill(0)
            direction_matrix.fill(0)
            print("loop1")
            for x in range(width):
                cost_matrix[x, 0] = x * max_disparity
                cost_matrix[0, x] = x * max_disparity

            for x in range(1, width):
                for d in range(1, width):
                    min1 = cost_matrix[x - 1, d - 1] + abs(int(img_left[y, x]) - int(img_right[y, d]))
                    min2 = cost_matrix[x - 1, d] + max_disparity
                    min3 = cost_matrix[x, d - 1] + max_disparity

                    cmin = min(min1, min2, min3)
                    cost_matrix[x, d] = cmin
                    print("loop2")
                    if cmin == min1:
                        direction_matrix[x, d] = 1
                    elif cmin == min2:
                        direction_matrix[x, d] = 2
                    else:
                        direction_matrix[x, d] = 3

            x = width - 1
            d = width - 1

            while x > 0 and d > 0:
                print("loop3")
                if direction_matrix[x, d] == 1:
                    disparity_map[y, x] = abs(x - d)
                    x -= 1
                    d -= 1
                elif direction_matrix[x, d] == 2:
                    x -= 1
                elif direction_matrix[x, d] == 3:
                    d -= 1

        # cv2.imshow('Disparity Map', disparity_map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plt.imshow(disparity_map, cmap="gray")
        plt.colorbar(label="Magnitude")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Fourier Transform Magnitude")
        plt.show()
        time.sleep()


class corner:
    def __init__(self):
        pass

    def nms(self, matrix, kernel_size=5):
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        maxed_matrix = cv2.dilate(matrix, structuring_element)
        result_matrix = matrix.copy()
        result_matrix[matrix != maxed_matrix] = 0
        return result_matrix

    def detect_corners(self, image, window_size, k_value, threshold):
        img_copy = image.copy()
        height = image.shape[0]
        width = image.shape[1]
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        Ix = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
        IxIx = Ix ** 2
        IyIy = Iy ** 2
        IxIy = Ix * Iy

        offset = window_size // 2
        R_matrix = np.zeros((height, width))
        
        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                M_00 = np.sum(IxIx[y-offset:y+offset+1, x-offset:x+offset+1])
                M_11 = np.sum(IyIy[y-offset:y+offset+1, x-offset:x+offset+1])
                M_01 = np.sum(IxIy[y-offset:y+offset+1, x-offset:x+offset+1])
                M = np.array([[M_00, M_01], [M_01, M_11]])
                R = np.linalg.det(M) - k_value * (np.trace(M) ** 2)
                R_matrix[y, x] = R
        
        cv2.normalize(R_matrix, R_matrix, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        is_corner = R_matrix > threshold
        R_matrix[~is_corner] = 0
        R_matrix = self.nms(R_matrix, 10)  # Set kernel_size between 10-20 for better results
        
        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                if R_matrix[y, x] != 0:
                    cv2.circle(img_copy, (x, y), 1, (0, 0, 255), -1)
        
        return img_copy


def bilt_in():
    img = cv2.imread("D://computer_vision//tsukuba_r.png")
    #img = cv2.imread("D://computer_vision//1234.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 3, 5, 0.04)
    corners = cv2.dilate(corners, None)
    img[corners > 0.01 * corners.max()]=[0, 0, 255]
    cv2.imshow('Image with Corners', img)

# Load the image8
if __name__ == "__main__":
    #image_path = "your//image//path" 
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_left = cv2.imread("D://computer_vision//tsukuba_r.png")
    #img_left = cv2.imread("D://computer_vision//1234.png")
    img_right = cv2.imread("D://computer_vision//tsukuba_r.png", cv2.IMREAD_GRAYSCALE)
    bilt_in()
    # test = ThreeVison
    # test.normalStero_matching(img_left,img_right,16)
    test = corner()
    
    corner_image = test.detect_corners(img_left, 5, 0.04, 0.24)
    plt.imshow(corner_image[:, :, ::-1])
    plt.show()
