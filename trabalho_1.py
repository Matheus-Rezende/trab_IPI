import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Variáveis globais
imagem_carregada = None
botoes = []
fator = 1.0

# Funções para as opções do menu
def carregar_imagem():
    global imagem_carregada, botoes

    filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")])
    if filepath:
        # Carrega a imagem selecionada usando matplotlib
        imagem = plt.imread(filepath)

        # Converta a imagem OpenCV em um objeto PIL Image
        imagem_pil = Image.fromarray(imagem)

        # Converte a imagem em um formato suportado pelo Tkinter
        imagem_tk = ImageTk.PhotoImage(imagem_pil)
        
        # Exibe a imagem no canvas
        canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
        canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

        # Salva a imagem carregada globalmente
        imagem_carregada = imagem
        print(imagem_carregada)

def salvar_imagem():
    filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Imagens PNG", "*.png")])
    if filepath:
        # Implemente a lógica para salvar a imagem aqui
        print("Salvar imagem:", filepath)

def mostrar_controles_de_tamanho():

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    # Adiciona um botão de zoom
    botao_zoom_mais = tk.Button(canvas, text="Aumentar", command=tamanho_mais)
    botao_zoom_mais_tela = canvas.create_window(680, 50, anchor=tk.NW, window=botao_zoom_mais)
    botoes.append(botao_zoom_mais_tela)

    # Adiciona um botão de zoom
    botao_zoom_menos = tk.Button(canvas, text="Diminuir", command=tamanho_menos)
    botao_zoom_menos_tela = canvas.create_window(600, 50, anchor=tk.NE, window=botao_zoom_menos)
    botoes.append(botao_zoom_menos_tela)

    # botao_zoom_menos = tk.Button(janela_principal, text="-", height= 2, width=5,command=zoom_menos)
    # botao_zoom_menos.pack(side="left")

def mostrar_controles_de_brilho():

    # Limpa os botões anteriores, se houver
    limpar_botoes()
    
     # Adiciona um botão de brilho
    botao_brilho_mais = tk.Button(canvas, text="Brilho +", command=brilho_mais)
    botao_brilho_mais_tela = canvas.create_window(680, 50, anchor=tk.NW, window=botao_brilho_mais)
    botoes.append(botao_brilho_mais_tela)

    # Adiciona um botão de zoom
    botao_brilho_menos = tk.Button(canvas, text="Brilho -", command=brilho_menos)
    botao_brilho_menos_tela = canvas.create_window(600, 50, anchor=tk.NE, window=botao_brilho_menos)
    botoes.append(botao_brilho_menos_tela)

def brilho_mais():
    global imagem_carregada, fator
    
    # diminui o brilho em 10%
    fator = fator + 0.1
    imagem_ajustada = np.clip(imagem_carregada * fator, 0, 255).astype(np.uint8)
    
    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_ajustada)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo


def brilho_menos():
    global imagem_carregada, fator
    
    # diminui o brilho em 10%
    fator = fator - 0.1
    imagem_ajustada = np.clip(imagem_carregada * fator, 0, 255).astype(np.uint8)
    
    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_ajustada)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

def tamanho_mais():
    global imagem_carregada, fator
    '''
      img: imagem de entrada
      sch: fator de escala na altura
      scw: fator de escala na largura
    '''

    # Usando VIZINHO MAIS PROXIMO, POR ENQUANTO....
    if(imagem_carregada.ndim == 2):
        h, w = imagem_carregada.shape
    else:
        h, w, c = imagem_carregada.shape # dimensoes de img (linhas, colunas, planos de cor)

    # Aumentando o tamanho da imagem em 10%
    fator = fator + 0.1

    # aloca a nova imagem
    nh = int(round(h * fator))
    nw = int(round(w * fator))
    if(imagem_carregada.ndim == 2):
        newImg = np.zeros((nh, nw), dtype=np.uint8)
    else:
        newImg = np.zeros((nh, nw, c), dtype=np.uint8)

    # indices dos pixels da nova imagem
    Ro = np.arange(nh)
    Co = np.arange(nw)

    # calcula os fatores de escala
    Sr = float(h) / float(nh) # h = numero de linhas da imagem original; nh= nova
    Sc = float(w) / float(nw) # w = numero de colunas da imagem original; nw = nova

    #calcula o mapeamento dos indices
    Rm = np.floor(Ro * Sr).astype(int)
    Cm = np.floor(Co * Sc).astype(int)

    coord_new = [(x,y) for x in Ro for y in Co] # todas as coodenadas de pixel da imagem nova
    coord_ori = [(x,y) for x in Rm for y in Cm] # todos as coordendas novas mapeadas para a original
    for cn, co in zip(coord_new, coord_ori):
        newImg[cn] = imagem_carregada[co]
    
    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(newImg)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

def tamanho_menos():
    global imagem_carregada, fator
    '''
      img: imagem de entrada
      sch: fator de escala na altura
      scw: fator de escala na largura
    '''

    # Usando VIZINHO MAIS PROXIMO, POR ENQUANTO....
    if(imagem_carregada.ndim == 2):
        h, w = imagem_carregada.shape
    else:
        h, w, c = imagem_carregada.shape # dimensoes de img (linhas, colunas, planos de cor)

    # Aumentando o tamanho da imagem em 10%
    fator = fator - 0.1

    # aloca a nova imagem
    nh = int(round(h * fator))
    nw = int(round(w * fator))
    if(imagem_carregada.ndim == 2):
        newImg = np.zeros((nh, nw), dtype=np.uint8)
    else:
        newImg = np.zeros((nh, nw, c), dtype=np.uint8)

    # indices dos pixels da nova imagem
    Ro = np.arange(nh)
    Co = np.arange(nw)

    # calcula os fatores de escala
    Sr = float(h) / float(nh) # h = numero de linhas da imagem original; nh= nova
    Sc = float(w) / float(nw) # w = numero de colunas da imagem original; nw = nova

    #calcula o mapeamento dos indices
    Rm = np.floor(Ro * Sr).astype(int)
    Cm = np.floor(Co * Sc).astype(int)

    coord_new = [(x,y) for x in Ro for y in Co] # todas as coodenadas de pixel da imagem nova
    coord_ori = [(x,y) for x in Rm for y in Cm] # todos as coordendas novas mapeadas para a original
    for cn, co in zip(coord_new, coord_ori):
        newImg[cn] = imagem_carregada[co]
    
    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(newImg)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo
    
def negativo():
    global imagem_carregada

    imagem_negativa = 255 - imagem_carregada

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_negativa)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

def filter_box():
    global imagem_carregada

    box_kernelx = (1.0/5) * np.ones(5)
    box_kernely = (1.0/5) * np.ones(5)

    img_saida = cv2.sepFilter2D(imagem_carregada, -1, box_kernelx, box_kernely)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(img_saida)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

def gaussiano():

    print("Normalização")

def sobel():
    global imagem_carregada

    sobX = np.array([[-1,-2,-1], [0,0,0],[1,2,1]])
    sobY = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])

    Gx = cv2.filter2D(imagem_carregada, cv2.CV_64F, sobX) # gradiente na direção X (linhas)
    Gy = cv2.filter2D(imagem_carregada, cv2.CV_64F, sobY) # gtadente na direção Y (colunas)

    mag = np.sqrt(Gx**2 + Gy**2) #magnitude do vetor gradiente

    img_agucada = imagem_carregada + 0.4 * mag
    img_agucada[img_agucada > 255] = 255
    img_agucada = img_agucada.astype(np.uint8)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(img_agucada)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

def vizinho_mais_proximo():
    print("Vizinho mais proximo")

def bilinear():
    print("bilinear")  

def bicubico():
    print("bicubico")


# Função para limpar os botões de zoom
def limpar_botoes():
    global botoes
    for botao_window in botoes:
        canvas.delete(botao_window)

    for botao_window in botoes:
        canvas.delete(botao_window)
    botoes = []

def zoom():
    # Implemente a lógica para a opção "Zoom" aqui
    print("Zoom")

def brilho():
    print("Brilho")

def transformacao_de_intensidade():
    limpar_botoes()
    # Implemente a lógica para a opção "Transformação de Intensidade" aqui
    print("Transformação de Intensidade")

def filtragem_espacial():
    # Implemente a lógica para a opção "Filtragem Espacial" aqui
    print("Filtragem Espacial")

# Cria uma instância da janela principal
janela_principal = tk.Tk()
janela_principal.title("Minha Aplicação Tkinter")
janela_principal.geometry("1280x720")

# Cria um canvas para exibir a imagem
canvas = tk.Canvas(janela_principal, width=1280, height=720)
canvas.pack()

# Cria um menu principal
menu_principal = tk.Menu(janela_principal)
janela_principal.config(menu=menu_principal)

# Cria um rótulo para exibir a imagem
label_imagem = tk.Label(janela_principal)
label_imagem.pack()

# Cria um menu "Arquivo" com as opções "Carregar imagem" e "Salvar imagem"
menu_arquivo = tk.Menu(menu_principal, tearoff=0)
menu_principal.add_cascade(label="Arquivo", menu=menu_arquivo)
menu_arquivo.add_command(label="Carregar imagem", command=carregar_imagem)
menu_arquivo.add_command(label="Salvar imagem", command=salvar_imagem)

# Cria um menu "Editar" com a opção "Recursos de edição"
menu_editar = tk.Menu(menu_principal, tearoff=0)
menu_principal.add_cascade(label="Editar", menu=menu_editar)

# Adiciona submenus de "Recursos de edição"
submenu_recursos = tk.Menu(menu_editar, tearoff=0)
menu_editar.add_cascade(label="Recursos de edição", menu=submenu_recursos)
submenu_recursos.add_command(label="Redimensionamento", command=mostrar_controles_de_tamanho)
submenu_recursos.add_command(label="Brilho", command=mostrar_controles_de_brilho)
submenu_recursos.add_command(label="Transformação de Intensidade", command=transformacao_de_intensidade)
submenu_recursos.add_command(label="Filtragem Espacial", command=filtragem_espacial)

# Adiciona submenus de "Redimensionar"
submenu_redimensionar = tk.Menu(submenu_recursos, tearoff=0)
submenu_recursos.add_cascade(label="Reamostragens", menu=submenu_redimensionar)
submenu_redimensionar.add_command(label="Vizinho mais próximo", command=vizinho_mais_proximo)
submenu_redimensionar.add_command(label="Bilinear", command=bilinear)
submenu_redimensionar.add_command(label="Bicubico", command=bicubico)

# Adiciona submenus de "Efeitos"
submenu_efeitos = tk.Menu(submenu_recursos, tearoff=0)
submenu_recursos.add_cascade(label="Efeitos", menu=submenu_efeitos)
submenu_efeitos.add_command(label="Negativo", command=negativo)

# Adiciona submenus de "Suavização"
submenu_suavizacao = tk.Menu(submenu_recursos, tearoff=0)
submenu_recursos.add_cascade(label="Algoritmos de suavização", menu=submenu_suavizacao)
submenu_suavizacao.add_command(label="Filter Box", command=filter_box)
submenu_suavizacao.add_command(label="Gaussiano", command=gaussiano)

# Adiciona submenus de "Aguçamento"
submenu_agucamento = tk.Menu(submenu_recursos, tearoff=0)
submenu_recursos.add_cascade(label="Algoritmos de aguçamento", menu=submenu_agucamento)
submenu_agucamento.add_command(label="Sobel", command=sobel)







# Inicia o loop principal da aplicação
janela_principal.mainloop()
