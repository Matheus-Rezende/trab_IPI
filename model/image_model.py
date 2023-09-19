import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageModel:
    def __init__(self):
        self.image = None
        self.fator = 1.0
        self.gamma = 1.0

    def load_image(self, filepath):
        # Carregar a imagem do arquivo
        self.image = cv2.imread(filepath)

    def save_image(self, filepath):
        # Salvar a imagem
        cv2.imwrite(filepath, self.image)

    def apply_brightness(self, constant):
        # Aplicar ajuste de brilho
        self.image = cv2.add(self.image, np.ones(self.image.shape, dtype=np.uint8) * constant)

    def apply_resize(self, width, height):
        # Redimensionar a imagem
        self.image = cv2.resize(self.image, (width, height), interpolation=cv2.INTER_LINEAR_EXACT)

    def apply_negative(self):
        # Aplicar efeito negativo
        self.image = 255 - self.image

    def apply_filter_box(self):
        # Aplicar filtro de caixa
        box_kernelx = (1.0/5) * np.ones(5)
        box_kernely = (1.0/5) * np.ones(5)
        self.image = cv2.sepFilter2D(self.image, -1, box_kernelx, box_kernely)

    def apply_gaussian(self):
        # Aplicar filtro gaussiano
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)

    # Adicione métodos para outras transformações, como Laplaciano, Sobel, etc.

    def apply_gamma_correction(self):
        # Aplicar correção gamma
        c = 255 / (255 ** self.gamma)
        self.image = c * (self.image.astype(np.float64) ** self.gamma)

    def get_image_pil(self):
        # Converter a imagem OpenCV em um objeto PIL Image
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
