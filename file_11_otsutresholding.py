import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar grayscale
img_path = 'objekjelas.jpeg' # Ganti dengan path gambar Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Terapkan Global Thresholding (untuk perbandingan)
    ret_manual, thresh_manual = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Terapkan Otsu's Thresholding
    # Perhatikan bendera cv2.THRESH_OTSU
    ret_otsu, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Nilai 'ret' akan berisi ambang batas yang ditemukan oleh Otsu

    # Tampilkan gambar dan histogramnya
    plt.figure(figsize=(18, 6))

    plt.subplot(2, 3, 1)
    plt.title('Gambar Asli')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title(f'Global Threshold (T=127)')
    plt.imshow(thresh_manual, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title(f'Otsu\'s Threshold (T={ret_otsu:.0f})')
    plt.imshow(thresh_otsu, cmap='gray')
    plt.axis('off')

    # Tampilkan Histogram
    plt.subplot(2, 1, 2) # Mengambil 1 baris penuh di bawah untuk histogram
    plt.hist(img.flatten(), 256, [0, 256], color='gray')
    plt.axvline(127, color='red', linestyle='dashed', linewidth=1, label='Manual Threshold')
    plt.axvline(ret_otsu, color='blue', linestyle='solid', linewidth=2, label='Otsu Threshold')
    plt.title('Histogram Gambar')
    plt.xlabel('Intensitas Piksel')
    plt.ylabel('Frekuensi')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()