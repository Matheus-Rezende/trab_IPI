import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageView:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller

        self.root.title("Photoshop simplificado")
        self.root.geometry("1280x720")

        # Crie um canvas para exibir a imagem
        self.canvas = tk.Canvas(root, width=1280, height=720)
        self.canvas.pack()

        # Crie um menu principal
        self.main_menu = tk.Menu(root)
        root.config(menu=self.main_menu)

        # Crie um menu "Arquivo" com as opções "Carregar imagem" e "Salvar imagem"
        self.file_menu = tk.Menu(self.main_menu, tearoff=0)
        self.main_menu.add_cascade(label="Arquivo", menu=self.file_menu)
        self.file_menu.add_command(label="Carregar imagem", command=self.load_image)
        self.file_menu.add_command(label="Salvar imagem", command=self.save_image)

        # Cria um menu "Editar" com a opção "Recursos de edição"
        self.menu_edit = tk.Menu(self.main_menu, tearoff=0)
        self.main_menu.add_cascade(label="Editar", menu=self.menu_edit)

        # Adiciona submenus de "Recursos de edição"
        submenu_resources = tk.Menu(self.menu_edit, tearoff=0)
        self.menu_edit.add_cascade(label="Recursos de edição", menu=submenu_resources)

        submenu_resources.add_command(label="Redimensionamento", command=self.resize_image)

        submenu_resources.add_command(label="Brilho", command=None)

        # Adiciona submenus de "Filtros"
        submenu_filters = tk.Menu(submenu_resources, tearoff=0)
        submenu_resources.add_cascade(label="Filtros", menu=submenu_filters)
        submenu_filters.add_command(label="Gaussiano", command=None)
        submenu_filters.add_command(label="Filtro Box", command=None)
        submenu_filters.add_command(label="Mediana", command=None)
        submenu_filters.add_command(label="Laplaciano", command=None)
        submenu_filters.add_command(label="Sobel", command=None)
        submenu_filters.add_command(label="Aguçamento via Gradiente", command=None)
        submenu_filters.add_command(label="Canny", command=None)
        submenu_filters.add_command(label="Hough", command=None)

        # Adiciona submenus de "Efeitos"
        submenu_effects = tk.Menu(submenu_resources, tearoff=0)
        submenu_resources.add_cascade(label="Efeitos", menu=submenu_effects)
        submenu_effects.add_command(label="Negativo", command=None)

        # Adiciona submenus de "Histograma"
        submenu_histogram = tk.Menu(submenu_resources, tearoff=0)
        submenu_resources.add_cascade(label="Histograma", menu=submenu_histogram)
        submenu_histogram.add_command(label="Especificação", command=None)
        submenu_histogram.add_command(label="Equalização", command=None)

        # Correções de contraste
        submenu_contrast = tk.Menu(submenu_resources, tearoff=0)
        submenu_resources.add_cascade(label="Correções de contraste", menu=submenu_contrast)
        submenu_contrast.add_command(label="Logaritmica", command=None)
        submenu_contrast.add_command(label="Exponencial", command=None)
        submenu_contrast.add_command(label="Gamma", command=None)

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png")])
        if filepath:
            self.controller.load_image(filepath)
            self.update_image()

    def save_image(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("Imagens", "*.jpg *.png")])
        if filepath:
            self.controller.save_image(filepath)

    def update_image(self):
        image_pil = self.controller.get_image_pil()
        self.photo = ImageTk.PhotoImage(image=image_pil)
        self.canvas.create_image(320, 0, anchor=tk.NW, image=self.photo)

    def adjust_brightness(self):
        constant = 50  # Valor de exemplo, você pode permitir que o usuário defina isso
        self.controller.apply_brightness(constant)
        self.update_image()

    def apply_negative(self):
        self.controller.apply_negative()
        self.update_image()

    def apply_box_filter(self):
        self.controller.apply_filter_box()
        self.update_image()

    def apply_gaussian_filter(self):
        self.controller.apply_gaussian()
        self.update_image()

    def gamma_correction(self):
        gamma = 1.5  # Valor de exemplo, você pode permitir que o usuário defina isso
        self.controller.apply_gamma_correction(gamma)
        self.update_image()

    def resize_image(self):
        def exec_button(mult):
            self.controller.apply_resize(mult)
            self.update_image()
        
        mult = 0.5
        
        exec_button(mult)
        


