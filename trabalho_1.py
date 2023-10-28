import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Variáveis globais
imagem_carregada = None
botoes = []
campos = []

fator = 1
gamma = 1.0

# Funções para as opções do menu
def carregar_imagem():
    global imagem_carregada, botoes, fator, gamma

    filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")])
    if filepath:

        # Reseta valores 
        fator = 1.0
        gamma = 1.0
        # Carrega a imagem selecionada usando matplotlib
        imagem = cv2.imread(filepath)

        imagem_out = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # Converta a imagem OpenCV em um objeto PIL Image
        imagem_pil = Image.fromarray(imagem_out)

        # Converte a imagem em um formato suportado pelo Tkinter
        imagem_tk = ImageTk.PhotoImage(imagem_pil)
        
        # Exibe a imagem no canvas
        canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
        canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

        # Salva a imagem carregada globalmente
        imagem_carregada = imagem
        print(imagem_carregada)

def salvar_imagem():
    global imagem_carregada
    
    # Solicita o local de salvamento da imagem
    filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Imagens PNG", "*.png"), ("Todos os arquivos", "*.*")])
    if filepath:

        imagem_out = cv2.cvtColor(imagem_carregada, cv2.COLOR_BGR2RGB)

        # Converta a imagem OpenCV em um objeto PIL Image
        imagem_pil = Image.fromarray(imagem_out)
        # Salva a imagem
        imagem_pil.save(filepath)

def mostrar_controles_de_tamanho():

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    # Adiciona um botão de zoom
    botao_zoom_mais = tk.Button(canvas, text="Aumentar", command=tamanho_mais)
    botao_zoom_mais_tela = canvas.create_window(10, 50, anchor=tk.NW, window=botao_zoom_mais)
    botoes.append(botao_zoom_mais_tela)

    # Adiciona um botão de zoom
    botao_zoom_menos = tk.Button(canvas, text="Diminuir", command=tamanho_menos)
    botao_zoom_menos_tela = canvas.create_window(150, 50, anchor=tk.NE, window=botao_zoom_menos)
    botoes.append(botao_zoom_menos_tela)

    # botao_zoom_menos = tk.Button(janela_principal, text="-", height= 2, width=5,command=zoom_menos)
    # botao_zoom_menos.pack(side="left")

def mostrar_controles_de_brilho():

    # Limpa os botões anteriores, se houver
    limpar_botoes()
    
     # Adiciona um botão de brilho
    botao_brilho_mais = tk.Button(canvas, text="Brilho +", command=brilho_mais)
    botao_brilho_mais_tela = canvas.create_window(10, 50, anchor=tk.NW, window=botao_brilho_mais)
    botoes.append(botao_brilho_mais_tela)

    # Adiciona um botão de zoom
    botao_brilho_menos = tk.Button(canvas, text="Brilho -", command=brilho_menos)
    botao_brilho_menos_tela = canvas.create_window(150, 50, anchor=tk.NE, window=botao_brilho_menos)
    botoes.append(botao_brilho_menos_tela)

def brilho_mais():
    global imagem_carregada, fator
    
    # Define o valor da constante de brilho (ajuste conforme necessário)
    constante_brilho = 20  # Aumentar o brilho em 20 unidades

    # Adicionar a constante para aumentar o brilho
    imagem_brilho_aumentado = cv2.add(imagem_carregada, np.ones(imagem_carregada.shape, dtype=np.uint8) * constante_brilho)

    imagem_out = cv2.cvtColor(imagem_brilho_aumentado, cv2.COLOR_BGR2RGB)

    
    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_brilho_aumentado

def brilho_menos():
    global imagem_carregada, fator
    
    # Define o valor da constante de brilho (ajuste conforme necessário)
    constante_brilho = 20  # Diminuir o brilho em 20 unidades

    # Adicionar a constante para diminuir o brilho
    imagem_brilho_diminuido = cv2.subtract(imagem_carregada, np.ones(imagem_carregada.shape, dtype=np.uint8) * constante_brilho)

    imagem_out = cv2.cvtColor(imagem_brilho_diminuido, cv2.COLOR_BGR2RGB)

    
    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_brilho_diminuido

def tamanho_mais():
    global imagem_carregada, fator

    # Calcula a nova largura e altura
    nova_largura = int(imagem_carregada.shape[1] + fator * 5.0)
    nova_altura = int(imagem_carregada.shape[0] + fator * 5.0)

    # Redimensiona a imagem
    imagem_aumentada = cv2.resize(imagem_carregada, (nova_largura, nova_altura),interpolation=cv2.INTER_LINEAR_EXACT)  

    imagem_out = imagem_aumentada.astype(np.uint8)
    
    img_out = cv2.cvtColor(imagem_out, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(img_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_aumentada


def tamanho_menos():
    global imagem_carregada, fator
    
    global imagem_carregada, fator

    # Calcula a nova largura e altura
    nova_largura = int(imagem_carregada.shape[1] - fator * 5.0)
    nova_altura = int(imagem_carregada.shape[0] - fator * 5.0)

    # Redimensiona a imagem
    imagem_aumentada = cv2.resize(imagem_carregada, (nova_largura, nova_altura),interpolation=cv2.INTER_LINEAR_EXACT)  

    imagem_out = imagem_aumentada.astype(np.uint8)
    
    img_out = cv2.cvtColor(imagem_out, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(img_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_aumentada
    
def negativo():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    imagem_negativa = 255 - imagem_carregada

    imagem_out = cv2.cvtColor(imagem_negativa, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_negativa

def filter_box():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    box_kernelx = (1.0/5) * np.ones(5)
    box_kernely = (1.0/5) * np.ones(5)

    img_saida = cv2.sepFilter2D(imagem_carregada, -1, box_kernelx, box_kernely)

    imagem_out = cv2.cvtColor(img_saida, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = img_saida

def gaussiano():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    imagem_suavizada = cv2.GaussianBlur(imagem_carregada, (5, 5), 0)

    imagem_out = cv2.cvtColor(imagem_suavizada, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_suavizada

def laplaciano():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    # Converter a imagem para escala de cinza (opcional)
    imagem_em_escala_de_cinza = cv2.cvtColor(imagem_carregada, cv2.COLOR_BGR2GRAY)

    # Aplicar o filtro Laplaciano
    imagem_filtrada = cv2.Laplacian(imagem_em_escala_de_cinza, cv2.CV_64F)

    # Converter a imagem filtrada de volta para inteiros de 8 bits sem sinal
    imagem_filtrada = cv2.convertScaleAbs(imagem_filtrada)


    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_filtrada)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_filtrada

def agucamento_gradiente():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    sobX = np.array([[-1,-2,-1], [0,0,0],[1,2,1]])
    sobY = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])

    Gx = cv2.filter2D(imagem_carregada, cv2.CV_64F, sobX) # gradiente na direção X (linhas)
    Gy = cv2.filter2D(imagem_carregada, cv2.CV_64F, sobY) # gtadente na direção Y (colunas)

    mag = np.sqrt(Gx**2 + Gy**2) #magnitude do vetor gradiente

    img_agucada = imagem_carregada + 0.4 * mag
    img_agucada[img_agucada > 255] = 255
    img_agucada = img_agucada.astype(np.uint8)

    imagem_out = cv2.cvtColor(img_agucada, cv2.COLOR_BGR2RGB)


    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = img_agucada

def sobel():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    # Aplicar o filtro Sobel em cada canal de cor
    sobel_x_r = cv2.Sobel(imagem_carregada[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_r = cv2.Sobel(imagem_carregada[:,:,0], cv2.CV_64F, 0, 1, ksize=3)

    sobel_x_g = cv2.Sobel(imagem_carregada[:,:,1], cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_g = cv2.Sobel(imagem_carregada[:,:,1], cv2.CV_64F, 0, 1, ksize=3)

    sobel_x_b = cv2.Sobel(imagem_carregada[:,:,2], cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_b = cv2.Sobel(imagem_carregada[:,:,2], cv2.CV_64F, 0, 1, ksize=3)

    # Calcular a magnitude do gradiente para cada canal
    magnitude_r = cv2.magnitude(sobel_x_r, sobel_y_r)
    magnitude_g = cv2.magnitude(sobel_x_g, sobel_y_g)
    magnitude_b = cv2.magnitude(sobel_x_b, sobel_y_b)

    # Normalizar a magnitude para a faixa de 0 a 255
    magnitude_r = cv2.normalize(magnitude_r, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_g = cv2.normalize(magnitude_g, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_b = cv2.normalize(magnitude_b, None, 0, 255, cv2.NORM_MINMAX)

    # Mesclar as magnitudes de volta para uma imagem colorida
    imagem_sobel = cv2.merge((magnitude_r, magnitude_g, magnitude_b))

    # Converter a imagem resultante para inteiros de 8 bits sem sinal
    imagem_sobel = cv2.convertScaleAbs(imagem_sobel)

    imagem_out = cv2.cvtColor(imagem_sobel, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_sobel

def canny():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    img_gray = cv2.cvtColor(imagem_carregada, cv2.COLOR_BGR2GRAY)

    # Extrai contorno
    img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
    canny_img = cv2.Canny(img_gray, 30, 150)
    # (contorno, _) = cv2.findContours(canny_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img_carregada_gray, contorno, -1, (0, 255, 0))
    
    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_canny_pil = Image.fromarray(canny_img)
    # imagem_carregada_pil = Image.fromarray(img_carregada_gray)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_canny_tk = ImageTk.PhotoImage(imagem_canny_pil)
    # imagem_carregada_tk = ImageTk.PhotoImage(imagem_carregada_pil)

    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_canny_tk)
    # canvas.create_image(280, 700, anchor=tk.NW, image=imagem_carregada_tk)

    canvas.imagem_canny_tk = imagem_canny_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo
    # canvas.imagem_carregada_tk = imagem_carregada_tk 

    # Salvando a alteração na imagem global
    imagem_carregada = canny_img 

def mediana():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    imagem_filtrada = cv2.medianBlur(imagem_carregada, 5)

    imagem_out = cv2.cvtColor(imagem_filtrada, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_filtrada

def especificacao_hist():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")])
    if filepath:

        segunda_imagem = cv2.imread(filepath)
        
        

    chans_img = cv2.split(imagem_carregada)
    chans_ref = cv2.split(segunda_imagem)
    print(segunda_imagem)

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
    

    img_out = np.zeros(imagem_carregada.shape) # imagem de saida

    for c in range(3):
        for i in range(256):
            diff = np.absolute(cdf_ref[:,c] - cdf_input[i,c])
            indice = diff.argmin()
            img_out[imagem_carregada[:,:,c] == i, c] = indice

    img_out = img_out.astype(np.uint8)

    imagem_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = img_out

def equalizacao_hist():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()
    
    R = imagem_carregada.shape[0]
    C = imagem_carregada.shape[1]

    #calculo do histograma normalizado (pr)
    hist = cv2.calcHist([imagem_carregada], [0], None, [256], [0, 256]) 
    pr = hist/(R*C)

    # cummulative distribution function (CDF)
    cdf = pr.cumsum()
    sk = 255 * cdf
    sk = np.round(sk)

    # criando a imagem de saída
    img_out = np.zeros(imagem_carregada.shape, dtype=np.uint8)
    for i in range(256):
        img_out[imagem_carregada == i] = sk[i]

    
    imagem_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = img_out


def mostrar_botao_segunda_imagem():

    # Limpa os botões anteriores, se houver
    limpar_botoes()
    
     # Adiciona um botão de brilho
    botao_selecionar_img = tk.Button(canvas, text="Segunda imagem", command=especificacao_hist)
    botao_selecionar_img_tela = canvas.create_window(10, 50, anchor=tk.NW, window=botao_selecionar_img)
    botoes.append(botao_selecionar_img_tela)
    
def correcao_logaritmica():
    global imagem_carregada

    c = 255 / np.log(256)
    imagem_corrigida = c * np.log(1 + imagem_carregada)
    imagem_corrigida = np.uint8(imagem_corrigida)
    
    imagem_out = cv2.cvtColor(imagem_corrigida, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_corrigida

def correcao_exponencial():
    global imagem_carregada

    # Limpa os botões anteriores, se houver
    limpar_botoes()

    c = 255 / np.log(256)
    imagem_corrigida = np.exp(imagem_carregada.astype(np.float64))**(1/c) - 1
    imagem_corrigida = np.uint8(imagem_corrigida)
    
    imagem_out = cv2.cvtColor(imagem_corrigida, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_corrigida

def mostrar_controles_gamma():

    # Adiciona um botão de gamma
    botao_gamma_mais = tk.Button(canvas, text="Gamma +", command=gamma_mais)
    botao_gamma_mais_tela = canvas.create_window(10, 50, anchor=tk.NW, window=botao_gamma_mais)
    botoes.append(botao_gamma_mais_tela)

   # Adiciona um botão de gamma
    botao_gamma_menos = tk.Button(canvas, text="Gamma -", command=gamma_menos)
    botao_gamma_menos_tela = canvas.create_window(150, 50, anchor=tk.NW, window=botao_gamma_menos)
    botoes.append(botao_gamma_menos_tela)


def gamma_mais():
    global imagem_carregada, gamma

    c = 255 / (255 ** (gamma + 0.1))
    imagem_corrigida = c * (imagem_carregada.astype(np.float64) ** (gamma + 0.1))
    imagem_corrigida = imagem_corrigida.astype(np.uint8)

    imagem_out = cv2.cvtColor(imagem_corrigida, cv2.COLOR_BGR2RGB)
    
    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo
       
    # Salvando a alteração na imagem global
    imagem_carregada = imagem_corrigida

def gamma_menos():
    global imagem_carregada, gamma

    c = 255 / (255 ** (gamma - 0.1))
    imagem_corrigida = c * (imagem_carregada.astype(np.float64) ** (gamma - 0.1))
    imagem_corrigida = imagem_corrigida.astype(np.uint8)

    imagem_out = cv2.cvtColor(imagem_corrigida, cv2.COLOR_BGR2RGB)

    # Converta a imagem OpenCV em um objeto PIL Image
    imagem_pil = Image.fromarray(imagem_out)

    # Exibe a imagem com o brilho ajustado no canvas
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    canvas.create_image(320, 0, anchor=tk.NW, image=imagem_tk)
    canvas.imagem_tk = imagem_tk  # Salva uma referência para evitar que a imagem seja destruída pela coleta de lixo

    # Salvando a alteração na imagem global
    imagem_carregada = imagem_corrigida


    """
    camadas = []
    camadas.append(imagem_corrigida)
    imagF = camadas[0]
    for camada in camadas[1:]:
        imagF = imagF + camada
    imagf = imagf / len(camadas)
    """

# Função para limpar os botões de zoom
def limpar_botoes():
    global botoes
    for botao_window in botoes:
        canvas.delete(botao_window)

    for botao_window in botoes:
        canvas.delete(botao_window)
    botoes = []

def limpar_campos_texto():
    global campos
    for campo_window in campos:
        canvas.delete(campo_window)

    for campo_window in campos:
        canvas.delete(campo_window)
    campos = []

# Função para fechar a aplicação
def fechar_aplicacao():
    janela_principal.destroy()

# Cria uma instância da janela principal
janela_principal = tk.Tk()
janela_principal.title("Photoshop simplificado")
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

# Inicializa a janela em modo fullScreen
janela_principal.attributes('-fullscreen', True)

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

# Adiciona submenus de "Filtros"
submenu_filtros = tk.Menu(submenu_recursos, tearoff=0)
submenu_recursos.add_cascade(label="Filtros", menu=submenu_filtros)
submenu_filtros.add_command(label="Gaussiano", command=gaussiano)
submenu_filtros.add_command(label="Filtro Box", command=filter_box)
submenu_filtros.add_command(label="Mediana", command=mediana)

submenu_filtros.add_command(label="Laplaciano", command=laplaciano)
submenu_filtros.add_command(label="Sobel", command=sobel)
submenu_filtros.add_command(label="Aguçamento via Gradiente", command=agucamento_gradiente)
submenu_filtros.add_command(label="Canny", command=canny)


# Adiciona submenus de "Efeitos"
submenu_efeitos = tk.Menu(submenu_recursos, tearoff=0)
submenu_recursos.add_cascade(label="Efeitos", menu=submenu_efeitos)
submenu_efeitos.add_command(label="Negativo", command=negativo)

# Adiciona submenus de "Histograma"
submenu_histograma = tk.Menu(submenu_recursos, tearoff=0)
submenu_recursos.add_cascade(label="Histograma", menu=submenu_histograma)
submenu_histograma.add_command(label="Especificação", command=mostrar_botao_segunda_imagem)
submenu_histograma.add_command(label="Equalização", command=equalizacao_hist)

# Correções de contraste
submenu_contraste = tk.Menu(submenu_recursos, tearoff=0)
submenu_recursos.add_cascade(label="Correções de contraste", menu=submenu_contraste)
submenu_contraste.add_command(label="Logaritmica", command=correcao_logaritmica)
submenu_contraste.add_command(label="Exponencial", command=correcao_exponencial)
submenu_contraste.add_command(label="Gamma", command=mostrar_controles_gamma)

# Cria um menu "Arquivo" com as opções "Carregar imagem" e "Salvar imagem"
menu_sair = tk.Menu(menu_principal, tearoff=0)
menu_principal.add_cascade(label="Ações", menu=menu_sair)
menu_sair.add_command(label="Sair", command=fechar_aplicacao)

# Inicia o loop principal da aplicação
janela_principal.mainloop()
