import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar
img_path = 'gambar_gelap.jpg' # Ganti dengan path gambar Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Normalisasi piksel ke rentang 0-1
    img_float = img.astype(np.float32) / 255.0

    # Nilai Gamma
    gamma_val_brighten = 0.5  # Membuat gambar lebih terang
    gamma_val_darken = 2.0    # Membuat gambar lebih gelap

    # Terapkan koreksi gamma
    gamma_brightened = np.power(img_float, gamma_val_brighten)
    gamma_darkened = np.power(img_float, gamma_val_darken)

    # Skalakan kembali ke 0-255 dan konversi ke uint8
    gamma_brightened = (gamma_brightened * 255).astype(np.uint8)
    gamma_darkened = (gamma_darkened * 255).astype(np.uint8)

    # Tampilkan gambar
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Gambar Asli')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f'Gamma Correction (gamma={gamma_val_brighten})')
    plt.imshow(gamma_brightened, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f'Gamma Correction (gamma={gamma_val_darken})')
    plt.imshow(gamma_darkened, cmap='gray')
    plt.axis('off')
    plt.show()
