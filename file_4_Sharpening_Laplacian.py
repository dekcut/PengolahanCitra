import cv2
import numpy as np
import matplotlib.pyplot as plt
# Muat gambar grayscale
img_path = 'gambar_buram.jpeg' # Ganti dengan path gambar Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Terapkan filter Laplacian
    # Pastikan ddepth adalah cv2.CV_64F untuk menghindari clipping negatif
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # Konversi hasil Laplacian kembali ke uint8
    laplacian_8bit = cv2.convertScaleAbs(laplacian)
    # Menambahkan hasil Laplacian ke gambar asli untuk sharpening
    # Karena Laplacian bisa menghasilkan nilai negatif, kita harus hati-hati.
    # Cara yang lebih baik adalah:
    # sharpened_img = cv2.addWeighted(img, 1.0, laplacian_8bit, 1.0, 0)
    # Namun, untuk efek sederhana, kita bisa coba:
    sharpened_img = cv2.add(img, laplacian_8bit) # Mungkin terlalu agresif
    # Atau, yang lebih umum adalah menggunakan Unsharp Masking (tidak Laplacian langsung)
    # Kita bisa simulasikan sharpening dengan subtracting blurred version from original
    
    # Pendekatan Unsharp Masking (Lebih umum dan lebih baik untuk sharpening)
    # 1. Blur gambar asli
    blurred = cv2.GaussianBlur(img, (0, 0), 3) # Ukuran kernel (0,0) berarti dihitung otomatis
                                                # sigmaX=3
    # 2. Hitung mask (original - blurred)
    mask = cv2.subtract(img, blurred)
    # 3. Tambahkan mask ke gambar asli (dengan faktor scaling)
    sharpened_unsharp = cv2.addWeighted(img, 1.0, mask, 1.5, 0) # 1.5 adalah faktor jumlah sharpening
    # Tampilkan gambar
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Gambar Asli (Agak Buram)')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Laplacian Output (Abs)')
    plt.imshow(laplacian_8bit, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Sharpened (Unsharp Masking)')
    plt.imshow(sharpened_unsharp, cmap='gray')
    plt.axis('off')
    plt.show()
