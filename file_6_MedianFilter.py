import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar grayscale
img_path = 'gambar_bersih.jpeg' # Ganti dengan path gambar bersih Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Tambahkan Salt-and-Pepper Noise (dari kode sebelumnya)
    s_vs_p = 0.5
    amount = 0.04
    noisy_sp_img = np.copy(img)
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_sp_img[coords[0], coords[1]] = 255
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_sp_img[coords[0], coords[1]] = 0

    # Terapkan Median Filter
    # Ukuran kernel harus ganjil, misal 3, 5, 7
    median_filtered_img = cv2.medianBlur(noisy_sp_img, 5)

    # Tampilkan gambar
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Gambar Asli (Bersih)')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Dengan Salt-and-Pepper Noise')
    plt.imshow(noisy_sp_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Setelah Median Filter (kernel 5x5)')
    plt.imshow(median_filtered_img, cmap='gray')
    plt.axis('off')
    plt.show()
