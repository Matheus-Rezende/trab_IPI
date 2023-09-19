class ImageController:
    def __init__(self, model):
        self.model = model

    def load_image(self, filepath):
        self.model.load_image(filepath)

    def save_image(self, filepath):
        self.model.save_image(filepath)

    def apply_brightness(self, constant):
        self.model.apply_brightness(constant)

    def apply_resize(self, width, height):
        self.model.apply_resize(width, height)

    def apply_negative(self):
        self.model.apply_negative()

    def apply_filter_box(self):
        self.model.apply_filter_box()

    def apply_gaussian(self):
        self.model.apply_gaussian()

    # Adicione métodos para outras transformações, como Laplaciano, Sobel, etc.

    def apply_gamma_correction(self):
        self.model.apply_gamma_correction()

    def get_image_pil(self):
        return self.model.get_image_pil()

