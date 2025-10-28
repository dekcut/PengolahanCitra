import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar grayscale (idealnya gambar dengan pencahayaan tidak merata)
img_path = 'cahaya.jpg' # Ganti dengan path gambar Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Terapkan Adaptive Thresholding - Mean
    # block_size = 11 (ukuran area piksel tetangga yang dipertimbangkan)
    # C = 2 (konstanta yang dikurangi dari rata-rata/gaussian)
    thresh_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

    # Terapkan Adaptive Thresholding - Gaussian
    thresh_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

    # Tampilkan gambar
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Gambar Asli (Pencahayaan Tidak Rata)')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adaptive Threshold (Mean)')
    plt.imshow(thresh_mean, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Adaptive Threshold (Gaussian)')
    plt.imshow(thresh_gaussian, cmap='gray')
    plt.axis('off')
    plt.show()