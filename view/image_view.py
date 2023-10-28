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
        self.menu_principal = tk.Menu(root)
        root.config(menu=self.menu_principal)

        # Crie um menu "Arquivo" com as opções "Carregar imagem" e "Salvar imagem"
        self.menu_arquivo = tk.Menu(self.menu_principal, tearoff=0)
        self.menu_principal.add_cascade(label="Arquivo", menu=self.menu_arquivo)
        self.menu_arquivo.add_command(label="Carregar imagem", command=self.load_image)
        self.menu_arquivo.add_command(label="Salvar imagem", command=self.save_image)

        # # Cria um menu "Editar" com a opção "Recursos de edição"
        # self.menu_editar = tk.Menu(self.menu_principal, tearoff=0)
        # self.menu_principal.add_cascade(label="Editar", menu=self.menu_editar)

        # # Adiciona submenus de "Recursos de edição"
        # self.submenu_recursos = tk.Menu(self.menu_editar, tearoff=0)
        # self.menu_editar.add_cascade(label="Recursos de edição", menu=self.submenu_recursos)

        # self.submenu_recursos.add_command(label="Redimensionamento", command=self.resize_image)
        # self.submenu_recursos.add_command(label="Brilho", command=self.adjust_brightness)

        # Crie botões e ligue-os aos métodos do Controller
        self.button_brightness = tk.Button(root, text="Ajustar Brilho", command=self.adjust_brightness)
        self.button_brightness.pack()
        self.button_negative = tk.Button(root, text="Efeito Negativo", command=self.apply_negative)
        self.button_negative.pack()
        self.button_box_filter = tk.Button(root, text="Filtro de Caixa", command=self.apply_box_filter)
        self.button_box_filter.pack()
        self.button_gaussian_filter = tk.Button(root, text="Filtro Gaussiano", command=self.apply_gaussian_filter)
        self.button_gaussian_filter.pack()
        self.button_gamma_correction = tk.Button(root, text="Correção Gamma", command=self.gamma_correction)
        self.button_gamma_correction.pack()
        self.button_resize = tk.Button(root, text="Redimensionar", command=self.resize_image)
        self.button_resize.pack()

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
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

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
        width = 640  # Valor de exemplo, você pode permitir que o usuário defina isso
        height = 480  # Valor de exemplo, você pode permitir que o usuário defina isso
        self.controller.apply_resize(width, height)
        self.update_image()
