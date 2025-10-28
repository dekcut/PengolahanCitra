import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar berwarna
img_path = 'apple.jpeg' # Ganti dengan path gambar berwarna Anda
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # OpenCV membaca gambar dalam format BGR, konversi ke RGB untuk Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Hitung Histogram Warna untuk setiap saluran ---
    colors = ('b', 'g', 'r') # Sesuai urutan BGR OpenCV

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title('Gambar Asli')
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Histogram Warna')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=f'{col.upper()} Channel')
    plt.xlim([0, 256])
    plt.xlabel('Intensitas Piksel')
    plt.ylabel('Frekuensi')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Untuk klasifikasi, Anda bisa menggabungkan histogram ini menjadi satu vektor fitur.
    # Misalnya, hist_b.flatten(), hist_g.flatten(), hist_r.flatten()
    # Atau, Anda bisa menggunakan histogram 2D atau 3D di ruang warna HSV/L*a*b*