# import cv2
# import numpy as np
# import unittest
# import matplotlib.pyplot as plt

# def hough_lines_acc(image, rho=1, theta=np.pi/180, threshold=100):
#     """
#     Apply Hough Transform to detect lines in an image and return the accumulator.
    
#     Parameters:
#     - image: numpy array
#         Edge-detected binary image.
#     - rho: float, optional (default=1)
#         Resolution of the accumulator in pixels.
#     - theta: float, optional (default=np.pi/180)
#         Angle resolution of the accumulator in radians.
#     - threshold: int, optional (default=100)
#         Accumulator threshold parameter. Only those lines are returned that get enough votes.
        
#     Returns:
#     - accumulator: numpy array
#         Hough Transform accumulator.
#     - thetas: numpy array
#         Array of theta values.
#     - rhos: numpy array
#         Array of rho values.
#     """
    
#     # Apply Hough Transform to detect lines
#     lines = cv2.HoughLines(image, rho, theta, threshold)
    
#     # Get the number of accumulator cells
#     diag_len = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
#     max_rho = diag_len
#     rhos = np.linspace(-max_rho, max_rho, max_rho * 2)
#     thetas = np.degrees(np.arange(-np.pi/2, np.pi/2, theta))
    
#     # Initialize the accumulator
#     accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)  # Updated dtype to int
    
#     if lines is not None:
#         for line in lines:
#             rho, theta = line[0]
#             theta = np.degrees(theta)
#             rho_idx = np.argmin(np.abs(rhos - rho))
#             theta_idx = np.argmin(np.abs(thetas - theta))
#             accumulator[rho_idx, theta_idx] += 1
            
#     return accumulator, thetas, rhos

# def display_image(image, title="Image"):
#     """
#     Display the image using matplotlib.
    
#     Parameters:
#     - image: numpy array
#         Image to be displayed.
#     - title: str, optional (default="Image")
#         Title of the displayed window.
#     """
#     plt.imshow(image, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# class TestHoughTransform(unittest.TestCase):
    
#     def test_hough_lines_acc(self):
#         # Load an image from file
#         image = cv2.imread('ola2.jpg', cv2.IMREAD_GRAYSCALE)
        
#         # Apply edge detection (Canny)
#         edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
#         # Apply Hough Transform and get the accumulator
#         accumulator, thetas, rhos = hough_lines_acc(edges)
        
#         # Check if the accumulator is not empty
#         self.assertTrue(np.any(accumulator))

# if __name__ == '__main__':
#     unittest.main()

# # You can uncomment the following lines to run the display function
# # image = cv2.imread('path_to_your_image.jpg', cv2.IMREAD_GRAYSCALE)
# # edges = cv2.Canny(image, 50, 150, apertureSize=3)
# # accumulator, thetas, rhos = hough_lines_acc(edges)
# # display_image(accumulator, title="Hough Transform")

# # # Load an image from file
# # image = cv2.imread('ola2.jpg', cv2.IMREAD_GRAYSCALE)

# # # Apply edge detection (Canny)
# # edges = cv2.Canny(image, 50, 150, apertureSize=3)

# # # Apply Hough Transform
# # hough_image = hough_transform(edges)

# # # Save or display the result
# # cv2.imwrite('hough_transform_result.jpg', hough_image)

# import cv2
# import numpy as np

# def hough_transform(image_path, rho=1, theta=np.pi/180, threshold=100):
#     # Carregar a imagem
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Detectar bordas usando Canny
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Aplicar a Transformada de Hough
#     lines = cv2.HoughLines(edges, rho, theta, threshold)
    
#     if lines is not None:
#         for line in lines:
#             rho, theta = line[0]
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
            
#             # Desenhar as linhas na imagem original
#             cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
#     # Salvar ou mostrar a imagem resultante
#     cv2.imshow('Hough Transform Result', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Caminho da sua imagem
# image_path = 'ola2.jpg'
# hough_transform(image_path)

# import cv2
# import numpy as np
# import unittest
# import matplotlib.pyplot as plt

# def draw_detected_lines(image_path, lines):
#     """
#     Draw the detected lines on the original image.
    
#     Parameters:
#     - image_path: str
#         Path to the original image.
#     - lines: list
#         Detected lines from the Hough Transform.
#     """
#     image = cv2.imread(image_path)
#     if lines is not None:
#         for line in lines:
#             rho, theta = line[0]
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#             cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title('Detected Lines')
#     plt.axis('off')
#     plt.show()

# # Load an image from file
# image_path = 'quadrado.jpg'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Apply edge detection (Canny)
# edges = cv2.Canny(image, 50, 150, apertureSize=3)

# # Apply Hough Transform to detect lines
# lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

# # Draw the detected lines on the original image
# draw_detected_lines(image_path, lines)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_detected_lines(image_path, lines):
    """
    Draw the detected lines on the original image.
    
    Parameters:
    - image_path: str
        Path to the original image.
    - lines: list
        Detected lines from the Hough Transform.
    """
    image = cv2.imread(image_path)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Lines')
    plt.axis('off')
    plt.show()

# Load an image from file
image_path = 'images.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply edge detection (Canny)
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Apply Probabilistic Hough Transform to detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# Draw the detected lines on the original image
draw_detected_lines(image_path, lines)
