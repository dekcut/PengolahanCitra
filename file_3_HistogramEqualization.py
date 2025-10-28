import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar grayscale
img_path = 'gambar_kontras_rendah.jpeg' # Ganti dengan path gambar Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Lakukan Histogram Equalization
    equalized_img = cv2.equalizeHist(img)

    # Hitung histogram
    hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

    # Tampilkan gambar dan histogramnya
    plt.figure(figsize=(15, 8))

    # Gambar Asli
    plt.subplot(2, 2, 1)
    plt.title('Gambar Asli')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Histogram Asli')
    plt.plot(hist_original, color='black')
    plt.xlim([0, 256])

    # Gambar yang Di-equalize
    plt.subplot(2, 2, 3)
    plt.title('Gambar Setelah Equalization')
    plt.imshow(equalized_img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Histogram Setelah Equalization')
    plt.plot(hist_equalized, color='black')
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()
