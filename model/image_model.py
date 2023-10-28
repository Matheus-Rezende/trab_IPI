from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageModel:
    def __init__(self):
        self._image = None # Original e imutavel
        self.image = None # Manipulavel

        self.fator = 1.0
        self.gamma = 1.0

    def load_image(self, filepath):
        # Carregar a imagem do arquivo
        self._image = cv2.imread(filepath)
        self.image = self._image.copy()


    def save_image(self, filepath):
        # Salvar a imagem
        image_out = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Converta a imagem OpenCV em um objeto PIL Image
        image_pil = Image.fromarray(image_out)

        # Salva a imagem
        image_pil.save(filepath)

    
    def reset_image(self):#############################################################################
        # Reseta imagem manipulada
        self.image = self._image.copy()


    def apply_brightness(self, constant):
        # Verificação para aumento ou diminuição do brilho
        if constant > 0:
            self.image = cv2.add(self.image, np.ones(self.image.shape, dtype=np.uint8) * constant)
        elif constant < 0:
            self.image = cv2.subtract(self.image, np.ones(self.image.shape, dtype=np.uint8) * constant)


    def apply_resize(self, mult): 
        # Calcula a nova largura e altura
        # se mult > 0 aumenta e se < 0 diminui
        if mult > 0:
            print(mult)
            print(self.image.shape[1])
            print(self.image.shape[0])
            width = int(self.image.shape[1] * self.fator * mult)
            height = int(self.image.shape[0] * self.fator * mult)
            print(width)
            print(height)

            # Redimensiona a imagem
            image_resize = cv2.resize(self.image, (width, height),interpolation=cv2.INTER_LINEAR_EXACT)  

            # Converte para uint8
            self.image = image_resize.astype(np.uint8)


    def apply_negative(self):
        # Aplicar efeito negativo
        self.image = 255 - self.image


    def apply_filter_box(self, kernel_size = 5):
        # Verifica o tamanho passado como parâmetro
        if kernel_size >= 3:
            # Verifica se o valor é par o impar
            if (kernel_size % 2) !=  0:
                _kernel_size = kernel_size
            else:
                _kernel_size = kernel_size - 1

            # Constroi kernel
            box_kernelx = (1.0/_kernel_size) * np.ones(_kernel_size)
            box_kernely = (1.0/_kernel_size) * np.ones(_kernel_size)

            # Filtra a imagem
            self.image = cv2.sepFilter2D(self.image, -1, box_kernelx, box_kernely)


    def apply_gaussian(self, kernel_size):
        # Verifica o tamanho passado como parâmetro
        if kernel_size >= 3:
            # Verifica se o valor é par o impar
            if (kernel_size % 2) !=  0:
                _kernel_size = kernel_size
            else:
                _kernel_size = kernel_size - 1

            # Aplicar filtro gaussiano
            self.image = cv2.GaussianBlur(self.image, (_kernel_size, _kernel_size), 0)


    def laplacian(self):
        # Converter a imagem para escala de cinza (opcional)
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Aplicar o filtro Laplaciano
        self.image = cv2.Laplacian(image_gray, cv2.CV_64F)

        # Converter a imagem filtrada de volta para inteiros de 8 bits sem sinal
        self.image = cv2.convertScaleAbs(self.image)


    def gradient_sharpening(self):
        sobX = np.array([[-1,-2,-1], [0,0,0],[1,2,1]])
        sobY = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])

        Gx = cv2.filter2D(self.image, cv2.CV_64F, sobX) # gradiente na direção X (linhas)
        Gy = cv2.filter2D(self.image, cv2.CV_64F, sobY) # gradiente na direção Y (colunas)

        mag = np.sqrt(Gx**2 + Gy**2) # magnitude do vetor gradiente

        img_gradient = self.image + 0.4 * mag
        img_gradient[img_gradient > 255] = 255

        self.image = img_gradient.astype(np.uint8)


    def sobel(self, k_size):
        if k_size >= 3:
            # Verifica se o valor é par o impar
            if (k_size % 2) !=  0:
                _k_size = k_size
            else:
                _k_size = k_size - 1

            # Aplicar o filtro Sobel em cada canal de cor
            sobel_x_r = cv2.Sobel(self.image[:,:,0], cv2.CV_64F, 1, 0, ksize = _k_size)
            sobel_y_r = cv2.Sobel(self.image[:,:,0], cv2.CV_64F, 0, 1, ksize = _k_size)

            sobel_x_g = cv2.Sobel(self.image[:,:,1], cv2.CV_64F, 1, 0, ksize = _k_size)
            sobel_y_g = cv2.Sobel(self.image[:,:,1], cv2.CV_64F, 0, 1, ksize = _k_size)

            sobel_x_b = cv2.Sobel(self.image[:,:,2], cv2.CV_64F, 1, 0, ksize = _k_size)
            sobel_y_b = cv2.Sobel(self.image[:,:,2], cv2.CV_64F, 0, 1, ksize = _k_size)

            # Calcular a magnitude do gradiente para cada canal
            magnitude_r = cv2.magnitude(sobel_x_r, sobel_y_r)
            magnitude_g = cv2.magnitude(sobel_x_g, sobel_y_g)
            magnitude_b = cv2.magnitude(sobel_x_b, sobel_y_b)

            # Normalizar a magnitude para a faixa de 0 a 255
            magnitude_r = cv2.normalize(magnitude_r, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_g = cv2.normalize(magnitude_g, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_b = cv2.normalize(magnitude_b, None, 0, 255, cv2.NORM_MINMAX)

            # Mesclar as magnitudes de volta para uma imagem colorida
            self.image = cv2.merge((magnitude_r, magnitude_g, magnitude_b))


    def canny(self, gaussian_blur, threshold1 = 30, threshold2 = 150):
        if gaussian_blur >= 3:
            # Verifica se o valor é par o impar
            if (gaussian_blur % 2) !=  0:
                _gaussian_blur = gaussian_blur
            else:
                _gaussian_blur = gaussian_blur - 1
                
            img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Extrai contorno
            img_gray = cv2.GaussianBlur(img_gray, (_gaussian_blur , _gaussian_blur), 0)
            self.image = cv2.Canny(img_gray, threshold1, threshold2)


    def median(self, ksize):
        if ksize >= 3:
            # Verifica se o valor é par o impar
            if (ksize % 2) !=  0:
                _ksize = ksize
            else:
                _ksize = ksize - 1

            self.image = cv2.medianBlur(self.image, _ksize)


    def specification_hist(self):
        filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")])

        if filepath:
            sec_image = cv2.imread(filepath)
            
        chans_img = cv2.split(self.image)
        chans_ref = cv2.split(sec_image)

        # iterage nos canais da imagem de entrada e calcula o histograma
        pr = np.zeros((256, 3))
        for chan, n in zip(chans_img, np.arange(3)):
            pr[:,n] = cv2.calcHist([chan], [0], None, [256], [0, 256]).ravel()

        # iterage nos canais da imagem de referencia e calcula o histograma
        pz = np.zeros((256, 3))
        for chan, n in zip(chans_ref, np.arange(3)):
            pz[:,n] = cv2.calcHist([chan], [0], None, [256], [0, 256]).ravel()
        
        # calcula as CDFs para a imagem de entrada
        cdf_input = np.zeros((256, 3))
        for i in range(3):
            cdf_input[:,i] = np.cumsum(pr[:,i]) # referencia
        
        # calcula as CDFs para a imagem de referencia
        cdf_ref = np.zeros((256,3))
        for i in range(3):
            cdf_ref[:,i] = np.cumsum(pz[:,i]) # referencia
        

        img_out = np.zeros(self.image.shape) # imagem de saida

        for c in range(3):
            for i in range(256):
                diff = np.absolute(cdf_ref[:,c] - cdf_input[i,c])
                indice = diff.argmin()
                img_out[self.image[:,:,c] == i, c] = indice

        self.image = img_out.astype(np.uint8)


    def equalization_hist(self):
        R = self.image.shape[0]
        C = self.image.shape[1]

        # calculo do histograma normalizado (pr)
        hist = cv2.calcHist([self.image], [0], None, [256], [0, 256]) 
        pr = hist/(R*C)

        # cummulative distribution function (CDF)
        cdf = pr.cumsum()
        sk = 255 * cdf
        sk = np.round(sk)

        # criando a imagem de saída
        img_out = np.zeros(self.image.shape, dtype=np.uint8)
        for i in range(256):
            img_out[self.image == i] = sk[i]
        
        self.image = img_out


    def logarithmic_correction(self):
        c = 255 / np.log(256)
        self.image = c * np.log(1 + self.image)
        self.image = np.uint8(self.image)


    def exponential_correction(self):
        c = 255 / np.log(256)
        self.image = np.exp(self.image.astype(np.float64))**(1/c) - 1
        self.image = np.uint8(self.image)


    def apply_gamma_correction(self, fator):
        # Aplicar correção gamma
        c = 255 / (255 ** (self.gamma + fator))
        self.image = c * (self.image.astype(np.float64) ** (self.gamma + fator))
        self.image = self.image.astype(np.uint8)


    def hough_transform(self):
        # Converte a imagem para tons de cinza
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Aplica a detecção de bordas usando o Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Aplica a transformada de Hough para detectar linhas
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        # Desenha as linhas detectadas na imagem original
        result = self.image.copy()
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        self.image = result


    def get_image_pil(self):
        # Converter a imagem OpenCV em um objeto PIL Image
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
