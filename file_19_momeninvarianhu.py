import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, img):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1. Buat sebuah citra biner sederhana
# Misalnya, bentuk persegi
img_persegi = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(img_persegi, (20, 20), (80, 80), 255, -1) # Persegi putih di latar hitam

# Misalnya, bentuk lingkaran
img_lingkaran = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(img_lingkaran, (50, 50), 30, 255, -1) # Lingkaran putih di latar hitam

# Contoh bentuk 'A' sederhana
img_A = np.zeros((100, 100), dtype=np.uint8)
pts_A = np.array([[30,70], [50,30], [70,70], [60,70], [55,55], [45,55], [40,70]], np.int32)
cv2.fillPoly(img_A, [pts_A], 255)
cv2.rectangle(img_A, (43, 58), (57, 65), 0, -1) # Lubang di tengah 'A'

# Tampilkan bentuk-bentuknya
display_image("Bentuk Persegi", img_persegi)
display_image("Bentuk Lingkaran", img_lingkaran)
display_image("Bentuk Huruf A", img_A)

# 2. Hitung momen untuk masing-masing bentuk
print("--- Momen Persegi ---")
moments_persegi = cv2.moments(img_persegi)
huMoments_persegi = cv2.HuMoments(moments_persegi)
print(f"Momen Hu (persegi):\n{huMoments_persegi.flatten()}")

print("\n--- Momen Lingkaran ---")
moments_lingkaran = cv2.moments(img_lingkaran)
huMoments_lingkaran = cv2.HuMoments(moments_lingkaran)
print(f"Momen Hu (lingkaran):\n{huMoments_lingkaran.flatten()}")

print("\n--- Momen Huruf A ---")
moments_A = cv2.moments(img_A)
huMoments_A = cv2.HuMoments(moments_A)
print(f"Momen Hu (huruf A):\n{huMoments_A.flatten()}")

# 3. Mari kita coba rotasi persegi dan lihat apakah momen Hu tetap sama
# Rotasi persegi 45 derajat
center = (50, 50)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_persegi = cv2.warpAffine(img_persegi, M, (100, 100))

display_image("Persegi Diputar 45 Derajat", rotated_persegi)

print("\n--- Momen Persegi Diputar ---")
moments_rotated_persegi = cv2.moments(rotated_persegi)
huMoments_rotated_persegi = cv2.HuMoments(moments_rotated_persegi)
print(f"Momen Hu (persegi diputar):\n{huMoments_rotated_persegi.flatten()}")

# Perhatikan bahwa nilai momen Hu sangat kecil, jadi kita bisa melihat perbedaannya
# lebih jelas jika kita mengambil logaritma absolutnya (seperti yang sering dilakukan)
print("\n--- Perbandingan Logaritma Absolut Momen Hu ---")
print("Log Hu Persegi Asli:")
print(-np.sign(huMoments_persegi) * np.log10(np.abs(huMoments_persegi)).flatten())
print("\nLog Hu Persegi Diputar:")
print(-np.sign(huMoments_rotated_persegi) * np.log10(np.abs(huMoments_rotated_persegi)).flatten())

# Anda akan melihat bahwa nilai-nilai ini sangat dekat, menunjukkan invarian rotasi
# Perbedaan kecil mungkin karena interpolasi selama rotasi.