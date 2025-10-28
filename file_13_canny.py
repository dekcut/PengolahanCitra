import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar grayscale
img_path = 'lotus.jpg' # Ganti dengan path gambar Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Terapkan Canny Edge Detector
    # low_threshold dan high_threshold adalah parameter kunci untuk hysteresis
    edges = cv2.Canny(img, 100, 200) # Contoh ambang batas: 100 dan 200

    # Tampilkan gambar
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Gambar Asli')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Canny Edges (T_low=100, T_high=200)')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.show()