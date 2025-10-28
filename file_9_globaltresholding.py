import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar grayscale
img_path = 'lotus.jpg' # Ganti dengan path gambar Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Terapkan Global Thresholding
    # Ambang batas T = 127
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Invert binary result (jika ingin objek putih, latar belakang hitam)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Tampilkan gambar
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Gambar Asli')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Global Threshold (T=127, BINARY)')
    plt.imshow(thresh1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Global Threshold (T=127, BINARY_INV)')
    plt.imshow(thresh2, cmap='gray')
    plt.axis('off')
    plt.show()