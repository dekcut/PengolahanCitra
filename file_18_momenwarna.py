import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar berwarna
img_path = 'apple.jpeg'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Pisahkan saluran BGR
    b, g, r = cv2.split(img)

    # --- Hitung Momen Warna untuk setiap saluran ---
    color_moments = []
    for channel in [b, g, r]:
        mean = np.mean(channel)
        variance = np.var(channel)
        
        # Skewness
        std_dev = np.sqrt(variance)
        if std_dev != 0:
            skewness = np.mean(((channel - mean) / std_dev)**3)
        else:
            skewness = 0 # Hindari pembagian dengan nol

        color_moments.extend([mean, variance, skewness])

    print(f"--- Fitur Warna (Momen Warna) ---")
    print(f"Momen Warna (Mean, Variance, Skewness per BGR channel):")
    print(f"  Blue: Mean={color_moments[0]:.2f}, Var={color_moments[1]:.2f}, Skew={color_moments[2]:.2f}")
    print(f"  Green: Mean={color_moments[3]:.2f}, Var={color_moments[4]:.2f}, Skew={color_moments[5]:.2f}")
    print(f"  Red: Mean={color_moments[6]:.2f}, Var={color_moments[7]:.2f}, Skew={color_moments[8]:.2f}")

    # Visualisasi gambar
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.title('Gambar Asli')
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()