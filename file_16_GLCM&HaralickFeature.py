import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Muat gambar grayscale
img_path = 'apple.jpeg' # Gambar yang menunjukkan tekstur buah
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Penting: GLCM bekerja paling baik pada gambar dengan level abu-abu yang lebih sedikit
    # Rescale intensitas ke 0-31 (32 level)
    img_rescaled = (img / 8).astype(np.uint8) # 256 / 8 = 32

    # Hitung GLCM
    # distances: Jarak piksel (misal, [1, 2])
    # angles: Sudut dalam radian (misal, [0, np.pi/4, np.pi/2, 3*np.pi/4] untuk 0, 45, 90, 135 derajat)
    # levels: Jumlah level abu-abu (32)
    # symmetric=True: GLCM simetris (i,j) == (j,i)
    # normed=True: GLCM dinormalisasi
    g = graycomatrix(img_rescaled, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                     levels=32, symmetric=True, normed=True)

    # Ekstrak Haralick Features dari GLCM
    contrast = graycoprops(g, 'contrast')
    dissimilarity = graycoprops(g, 'dissimilarity')
    homogeneity = graycoprops(g, 'homogeneity')
    energy = graycoprops(g, 'energy')
    correlation = graycoprops(g, 'correlation')
    # entropy: Perlu dihitung manual karena tidak ada di graycoprops secara langsung
    # Entropy = -sum(p * log2(p)) for p in GLCM entries (non-zero)

    print(f"--- Fitur Tekstur (GLCM - Haralick) ---")
    print(f"Contrast (avg across angles): {np.mean(contrast):.4f}")
    print(f"Dissimilarity (avg across angles): {np.mean(dissimilarity):.4f}")
    print(f"Homogeneity (avg across angles): {np.mean(homogeneity):.4f}")
    print(f"Energy (avg across angles): {np.mean(energy):.4f}")
    print(f"Correlation (avg across angles): {np.mean(correlation):.4f}")

    # Visualisasi
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Gambar Asli (Grayscale)')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gambar Rescaled (32 level)')
    plt.imshow(img_rescaled, cmap='gray')
    plt.axis('off')
    plt.show()