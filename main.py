import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen (cambia el nombre por tu imagen)
imagen = cv2.imread('imagen_fondo_de_ojo.jpg', cv2.IMREAD_COLOR)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# 1. Filtro de Promedio
promedio = cv2.blur(imagen, (5, 5))

# 2. Filtro Gaussiano
gaussiano = cv2.GaussianBlur(imagen, (5, 5), 0)

# 3. Filtro Laplaciano (realce)
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
laplaciano = cv2.Laplacian(gris, cv2.CV_64F)
laplaciano = cv2.convertScaleAbs(laplaciano)

# 4. Gradiente Sobel (realce)
sobelx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
gradiente = cv2.magnitude(sobelx, sobely)
gradiente = cv2.convertScaleAbs(gradiente)

# Mostrar resultados
titulos = ['Original', 'Promedio', 'Gaussiano', 'Laplaciano', 'Gradiente (Sobel)']
imagenes = [imagen_rgb, cv2.cvtColor(promedio, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(gaussiano, cv2.COLOR_BGR2RGB), laplaciano, gradiente]

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(imagenes[i], cmap='gray' if i >= 3 else None)
    plt.title(titulos[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
