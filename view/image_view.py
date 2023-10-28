import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageView:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.textfield = []
        self.button = []

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

        submenu_resources.add_command(label="Brilho", command=self.adjust_brightness)

        # Adiciona submenus de "Filtros"
        submenu_filters = tk.Menu(submenu_resources, tearoff=0)
        submenu_resources.add_cascade(label="Filtros", menu=submenu_filters)
        submenu_filters.add_command(label="Gaussiano", command=self.apply_gaussian_filter)
        submenu_filters.add_command(label="Filtro Box", command=self.apply_box_filter)
        submenu_filters.add_command(label="Mediana", command=self.median)
        submenu_filters.add_command(label="Laplaciano", command=self.laplacian)
        submenu_filters.add_command(label="Sobel", command=self.sobel)
        submenu_filters.add_command(label="Aguçamento via Gradiente", command=self.gradient_sharpening)
        submenu_filters.add_command(label="Canny", command=self.canny)
        submenu_filters.add_command(label="Hough", command=self.hough_transform)

        # Adiciona submenus de "Efeitos"
        submenu_effects = tk.Menu(submenu_resources, tearoff=0)
        submenu_resources.add_cascade(label="Efeitos", menu=submenu_effects)
        submenu_effects.add_command(label="Negativo", command=self.apply_negative)

        # Adiciona submenus de "Histograma"
        submenu_histogram = tk.Menu(submenu_resources, tearoff=0)
        submenu_resources.add_cascade(label="Histograma", menu=submenu_histogram)
        submenu_histogram.add_command(label="Especificação", command=self.specification_hist)
        submenu_histogram.add_command(label="Equalização", command=self.equalization_hist)

        # Correções de contraste
        submenu_contrast = tk.Menu(submenu_resources, tearoff=0)
        submenu_resources.add_cascade(label="Correções de contraste", menu=submenu_contrast)
        submenu_contrast.add_command(label="Logaritmica", command=self.logarithmic_correction)
        submenu_contrast.add_command(label="Exponencial", command=self.exponential_correction)
        submenu_contrast.add_command(label="Gamma", command=self.gamma_correction)

    def update_image(self):
        image_pil = self.controller.get_image_pil()
        self.photo = ImageTk.PhotoImage(image=image_pil)
        self.canvas.create_image(320, 0, anchor=tk.NW, image=self.photo)

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png")])
        if filepath:
            self.controller.load_image(filepath)
            self.update_image()

    def save_image(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("Imagens", "*.jpg *.png")])
        if filepath:
            self.controller.save_image(filepath)

    def adjust_brightness(self):
        def exec_button():
            constant = campo_texto.get()
            if constant:  # Verifique se a string não está vazia
                self.controller.apply_brightness(int(constant))
                self.update_image()

        # Cria uma variável StringVar
        campo_texto = tk.StringVar()

        campo_texto_entry = tk.Entry(self.canvas, textvariable=campo_texto)
        campo_texto_tela = self.canvas.create_window(10, 50, anchor=tk.NW, window=campo_texto_entry)
        self.textfield.append(campo_texto_tela)

        value = lambda: exec_button()

        botao = tk.Button(self.canvas, text="Ok", command=value)
        botao_tela = self.canvas.create_window(30, 50, anchor=tk.NW, window=botao)
        self.button.append(botao_tela)


    def resize_image(self):
        def exec_button():
            mult = campo_texto.get()
            if mult:  # Verifique se a string não está vazia
                self.controller.apply_resize(float(mult))
                self.update_image()

        # Cria uma variável StringVar
        campo_texto = tk.StringVar()

        campo_texto_entry = tk.Entry(self.canvas, textvariable=campo_texto)
        campo_texto_tela = self.canvas.create_window(10, 50, anchor=tk.NW, window=campo_texto_entry)
        self.textfield.append(campo_texto_tela)

        value = lambda: exec_button()

        botao = tk.Button(self.canvas, text="Ok", command=value)
        botao_tela = self.canvas.create_window(30, 50, anchor=tk.NW, window=botao)
        self.button.append(botao_tela)


    def apply_negative(self):
        self.controller.apply_negative()
        self.update_image()


    def apply_box_filter(self):
        def exec_button():
            kernel_size = campo_texto.get()
            if kernel_size:  # Verifique se a string não está vazia
                self.controller.apply_filter_box(int(kernel_size))
                self.update_image()

        # Cria uma variável StringVar
        campo_texto = tk.StringVar()

        campo_texto_entry = tk.Entry(self.canvas, textvariable=campo_texto)
        campo_texto_tela = self.canvas.create_window(10, 50, anchor=tk.NW, window=campo_texto_entry)
        self.textfield.append(campo_texto_tela)

        value = lambda: exec_button()

        botao = tk.Button(self.canvas, text="Ok", command=value)
        botao_tela = self.canvas.create_window(30, 50, anchor=tk.NW, window=botao)
        self.button.append(botao_tela)


    def apply_gaussian_filter(self):
        def exec_button():
            kernel_size = campo_texto.get()
            if kernel_size:  # Verifique se a string não está vazia
                self.controller.apply_gaussian(int(kernel_size))
                self.update_image()

        # Cria uma variável StringVar
        campo_texto = tk.StringVar()

        campo_texto_entry = tk.Entry(self.canvas, textvariable=campo_texto)
        campo_texto_tela = self.canvas.create_window(10, 50, anchor=tk.NW, window=campo_texto_entry)
        self.textfield.append(campo_texto_tela)

        value = lambda: exec_button()

        botao = tk.Button(self.canvas, text="Ok", command=value)
        botao_tela = self.canvas.create_window(30, 50, anchor=tk.NW, window=botao)
        self.button.append(botao_tela)


    def laplacian(self):
        self.controller.laplacian()
        self.update_image()


    def gradient_sharpening(self):
        self.controller.gradient_sharpening()
        self.update_image()


    def sobel(self):
        def exec_button():
            k_size = campo_texto.get()
            if k_size:  # Verifique se a string não está vazia
                self.controller.sobel(int(k_size))
                self.update_image()

        # Cria uma variável StringVar
        campo_texto = tk.StringVar()

        campo_texto_entry = tk.Entry(self.canvas, textvariable=campo_texto)
        campo_texto_tela = self.canvas.create_window(10, 50, anchor=tk.NW, window=campo_texto_entry)
        self.textfield.append(campo_texto_tela)

        value = lambda: exec_button()

        botao = tk.Button(self.canvas, text="Ok", command=value)
        botao_tela = self.canvas.create_window(30, 50, anchor=tk.NW, window=botao)
        self.button.append(botao_tela)


    def canny(self):
        def exec_button():
            gaussian_blur = campo_texto.get()
            if gaussian_blur:  # Verifique se a string não está vazia
                self.controller.canny(int(gaussian_blur))
                self.update_image()

        # Cria uma variável StringVar
        campo_texto = tk.StringVar()

        campo_texto_entry = tk.Entry(self.canvas, textvariable=campo_texto)
        campo_texto_tela = self.canvas.create_window(10, 50, anchor=tk.NW, window=campo_texto_entry)
        self.textfield.append(campo_texto_tela)

        value = lambda: exec_button()

        botao = tk.Button(self.canvas, text="Ok", command=value)
        botao_tela = self.canvas.create_window(30, 50, anchor=tk.NW, window=botao)
        self.button.append(botao_tela)


    def median(self):
        def exec_button():
            ksize = campo_texto.get()
            if ksize:  # Verifique se a string não está vazia
                self.controller.median(int(ksize))
                self.update_image()

        # Cria uma variável StringVar
        campo_texto = tk.StringVar()

        campo_texto_entry = tk.Entry(self.canvas, textvariable=campo_texto)
        campo_texto_tela = self.canvas.create_window(10, 50, anchor=tk.NW, window=campo_texto_entry)
        self.textfield.append(campo_texto_tela)

        value = lambda: exec_button()

        botao = tk.Button(self.canvas, text="Ok", command=value)
        botao_tela = self.canvas.create_window(30, 50, anchor=tk.NW, window=botao)
        self.button.append(botao_tela)


    def specification_hist(self):
        self.controller.specification_hist()
        self.update_image()


    def equalization_hist(self):
        self.controller.equalization_hist()
        self.update_image()


    def logarithmic_correction(self):
        self.controller.logarithmic_correction()
        self.update_image()


    def exponential_correction(self):
        self.controller.exponential_correction()
        self.update_image()


    def gamma_correction(self):
        def exec_button():
            gamma = campo_texto.get()
            if gamma:  # Verifique se a string não está vazia
                self.controller.apply_gamma_correction(float(gamma))
                self.update_image()

        # Cria uma variável StringVar
        campo_texto = tk.StringVar()

        campo_texto_entry = tk.Entry(self.canvas, textvariable=campo_texto)
        campo_texto_tela = self.canvas.create_window(10, 50, anchor=tk.NW, window=campo_texto_entry)
        self.textfield.append(campo_texto_tela)

        value = lambda: exec_button()

        botao = tk.Button(self.canvas, text="Ok", command=value)
        botao_tela = self.canvas.create_window(30, 50, anchor=tk.NW, window=botao)
        self.button.append(botao_tela)


    def hough_transform(self):
        self.controller.hough_transform()
        self.update_image()