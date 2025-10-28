import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(images, titles):
    """Fungsi pembantu untuk menampilkan beberapa gambar."""
    rows = 1
    cols = len(images)
    plt.figure(figsize=(cols * 5, 5))
    for i in range(cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 1. Muat citra (grayscale)
# Gunakan citra yang memiliki detail dan area mulus
try:
    img_path = 'cagrayscale.jpg' # Ganti dengan path ke gambar Anda
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {img_path}")
except FileNotFoundError as e:
    print(e)
    # Jika gambar tidak ditemukan, buat dummy image untuk demonstrasi
    print("Membuat citra dummy untuk demonstrasi...")
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1) # Persegi besar
    cv2.circle(img, (200, 200), 30, 255, -1) # Lingkaran
    cv2.line(img, (10, 10), (240, 240), 255, 3) # Garis diagonal
    img = cv2.GaussianBlur(img, (5,5), 0) # Sedikit blur untuk efek lebih realistis
    # Tambahkan noise untuk high-pass
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)


# Konversi ke float32 untuk perhitungan Fourier
img_float32 = np.float32(img)

# Tampilkan citra asli
display_images([img], ["Citra Asli"])

# 2. Lakukan DFT 2D
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)

# Geser komponen frekuensi nol ke tengah (untuk visualisasi yang lebih baik)
dft_shift = np.fft.fftshift(dft)

# Hitung spektrum magnitudo (untuk visualisasi)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
display_images([magnitude_spectrum], ["Spektrum Magnitudo (log)"])

# 3. Buat mask filter Low-Pass Gaussian
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2 # Pusat spektrum setelah fftshift

# Radius cut-off
d0_lp = 30 # Untuk low-pass (semakin besar, semakin sedikit blur)

# Buat masker Gaussian (Low-Pass)
mask_lp = np.zeros((rows, cols, 2), np.float32)
for i in range(rows):
    for j in range(cols):
        # Jarak dari pusat
        d = np.sqrt((i - crow)**2 + (j - ccol)**2)
        # Gaussian filter
        mask_lp[i,j,:] = np.exp(-(d**2) / (2 * d0_lp**2))

# 4. Terapkan filter Low-Pass
fshift_lp = dft_shift * mask_lp

# Geser kembali komponen frekuensi nol ke sudut (untuk IFFT)
f_ishift_lp = np.fft.ifftshift(fshift_lp)

# Lakukan IDFT untuk mendapatkan citra yang telah difilter
img_back_lp = cv2.idft(f_ishift_lp)
img_back_lp = cv2.magnitude(img_back_lp[:,:,0], img_back_lp[:,:,1]) # Ambil magnitudo
img_back_lp = cv2.normalize(img_back_lp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

display_images([img_back_lp], ["Citra Setelah Low-Pass (Gaussian)"])

# 5. Buat mask filter High-Pass Gaussian
# HPF adalah kebalikan dari LPF
mask_hp = 1 - mask_lp

# 6. Terapkan filter High-Pass
fshift_hp = dft_shift * mask_hp

# Geser kembali komponen frekuensi nol ke sudut (untuk IFFT)
f_ishift_hp = np.fft.ifftshift(fshift_hp)

# Lakukan IDFT untuk mendapatkan citra yang telah difilter
img_back_hp = cv2.idft(f_ishift_hp)
img_back_hp = cv2.magnitude(img_back_hp[:,:,0], img_back_hp[:,:,1]) # Ambil magnitudo
img_back_hp = cv2.normalize(img_back_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

display_images([img_back_hp], ["Citra Setelah High-Pass (Gaussian)"])

# Contoh Filtering Homomorfik (Konseptual, implementasi lengkap lebih kompleks)
# Logaritma citra
# log_img = np.log1p(img_float32 / 255.0) # Normalisasi dan log
# dft_log = cv2.dft(log_img, flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift_log = np.fft.fftshift(dft_log)

# Buat filter homomorfik (seringkali HPF yang dimodifikasi, misalnya Gaussian HPF yang dinaikkan)
# Misalnya: H_homo(u,v) = (Rh - Rl) * (1 - exp(-c * (D(u,v)^2 / D0^2))) + Rl
# Di mana Rh > 1 (high-pass gain), Rl < 1 (low-pass attenuation), D0 (cutoff frequency)
# mask_homo = np.zeros((rows, cols, 2), np.float32)
# R_h = 2.0 # High-frequency gain
# R_l = 0.5 # Low-frequency attenuation
# c = 1.0 # Sharpness of the filter
# D0_homo = 50.0 # Cutoff frequency

# for i in range(rows):
#     for j in range(cols):
#         D = np.sqrt((i - crow)**2 + (j - ccol)**2)
#         H = (R_h - R_l) * (1 - np.exp(-c * (D**2 / D0_homo**2))) + R_l
#         mask_homo[i,j,:] = H

# # Terapkan filter
# fshift_homo = dft_shift_log * mask_homo
# f_ishift_homo = np.fft.ifftshift(fshift_homo)
# img_back_homo_log = cv2.idft(f_ishift_homo)
# img_back_homo_log = cv2.magnitude(img_back_homo_log[:,:,0], img_back_homo_log[:,:,1])

# # Exponensial untuk kembali ke domain spasial
# img_back_homo = np.expm1(img_back_homo_log) # exp(x) - 1
# img_back_homo = cv2.normalize(img_back_homo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# display_images([img_back_homo], ["Citra Setelah Filtering Homomorfik"])