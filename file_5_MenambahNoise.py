import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar grayscale
img_path = 'gambar_bersih.jpeg' # Ganti dengan path gambar bersih Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # --- Tambahkan Gaussian Noise ---
    mean = 0
    var = 400 # Varian noise
    sigma = var**0.5
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
    noisy_gaussian_img = cv2.add(img, gaussian_noise) # Penjumlahan saturasi
    # Pastikan clipping 0-255 manual jika tidak pakai cv2.add dengan uint8
    noisy_gaussian_img = np.clip(img + gaussian_noise, 0, 255).astype(np.uint8)

    # --- Tambahkan Salt-and-Pepper Noise ---
    s_vs_p = 0.5 # Rasio salt vs pepper
    amount = 0.04 # Persentase piksel yang terpengaruh
    noisy_sp_img = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_sp_img[coords[0], coords[1]] = 255
    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_sp_img[coords[0], coords[1]] = 0

    # Tampilkan gambar-gambar dengan noise
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Gambar Asli')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Gaussian Noise')
    plt.imshow(noisy_gaussian_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Salt-and-Pepper Noise')
    plt.imshow(noisy_sp_img, cmap='gray')
    plt.axis('off')
    plt.show()
