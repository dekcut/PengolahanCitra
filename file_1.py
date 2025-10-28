import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar grayscale
img_path = 'anggur.jpeg' # Ganti dengan path gambar Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Parameter untuk kontras dan kecerahan
    alpha = 1.5  # Kontras (faktor pengali)
    beta = 30    # Kecerahan (nilai penambah)

    # Lakukan transformasi
    # Pastikan output tetap dalam rentang 0-255 dan tipe data uint8
    enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Tampilkan gambar
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Gambar Asli')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Enhanced (alpha={alpha}, beta={beta})')
    plt.imshow(enhanced_img, cmap='gray')
    plt.axis('off')
    plt.show()
