class ImageController:
    def __init__(self, model):
        self.model = model

    def load_image(self, filepath):
        self.model.load_image(filepath)

    def save_image(self, filepath):
        self.model.save_image(filepath)

    def reset_image(self):
        self.model.reset_image()

    def apply_brightness(self, constant):
        self.model.apply_brightness(constant)

    def apply_resize(self, mult):
        self.model.apply_resize(mult)

    def apply_negative(self):
        self.model.apply_negative()

    def apply_filter_box(self, kernel_size):
        self.model.apply_filter_box(kernel_size)

    def apply_gaussian(self, kernel_size):
        self.model.apply_gaussian(kernel_size)

    def laplacian(self):
        self.model.laplacian()

    def gradient_sharpening(self):
        self.model.gradient_sharpening()

    def sobel(self, k_size):
        self.model.sobel(k_size)

    def canny(self, gaussian_blur):
        self.model.canny(gaussian_blur)

    def median(self, ksize):
        self.model.median(ksize)

    def specification_hist(self):
        self.model.specification_hist()

    def equalization_hist(self):
        self.model.equalization_hist()

    def logarithmic_correction(self):
        self.model.logarithmic_correction()

    def exponential_correction(self):
        self.model.exponential_correction()

    def apply_gamma_correction(self):
        self.model.apply_gamma_correction()

    def hough_transform(self):
        self.model.hough_transform()

    def get_image_pil(self):
        return self.model.get_image_pil()

