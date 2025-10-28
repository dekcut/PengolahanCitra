import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def wiener_filter_deconvolution(img, psf, snr=None):
    """
    Implementasi sederhana Wiener Filter untuk dekonvolusi.
    Asumsi: Noise Gaussian aditif.
    
    Args:
        img (np.array): Gambar terdegradasi (grayscale).
        psf (np.array): Point Spread Function (kernel blur).
        snr (float): Signal-to-Noise Ratio. Jika None, estimasi dari gambar.
                     Nilai yang lebih besar berarti lebih sedikit noise.
                     Ini adalah K dalam beberapa literatur.
                     Biasanya dalam rentang 0.001 hingga 0.1 untuk gambar yang sangat noise,
                     hingga 1 atau 10 untuk gambar yang sedikit noise.
    Returns:
        np.array: Gambar yang direstorasi.
    """
    img_float = img.astype(np.float64) / 255.0
    psf_float = psf.astype(np.float64) / psf.sum() # Normalisasi PSF

    # Transformasi Fourier dari gambar terdegradasi dan PSF
    G = np.fft.fft2(img_float)
    H = np.fft.fft2(psf_float, s=img_float.shape) # PSF harus berukuran sama dengan gambar

    # Estimasi SNR jika tidak diberikan
    if snr is None:
        # Estimasi SNR adalah heuristik. Seringkali K = noise_var / signal_var
        # Atau bisa juga disimulasikan sebagai konstanta K.
        # Untuk demo, kita gunakan nilai heuristik atau default.
        K = 0.01 # Contoh nilai konstan untuk S_eta/S_f. Semakin kecil K, semakin agresif deblurring.
    else:
        K = 1 / snr # K = Noise_power / Signal_power = 1 / SNR

    # Wiener Filter
    # |H|^2 + K
    denom = np.abs(H)**2 + K
    
    # H* / (|H|^2 + K)
    wiener_filter = np.conj(H) / denom

    # F_hat = Wiener_Filter * G
    F_hat = wiener_filter * G

    # Inverse Fourier Transform
    restored_img = np.fft.ifft2(F_hat)
    restored_img = np.abs(restored_img) # Ambil magnitude karena bisa ada komponen imajiner kecil
    restored_img = np.clip(restored_img * 255, 0, 255).astype(np.uint8)
    
    return restored_img

# --- Simulasi Blur (Motion Blur sederhana) dan Noise ---
img_path = 'gambar_bersih.jpeg'
img_clean = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img_clean is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # 1. Buat PSF (Point Spread Function) untuk motion blur
    # Ini merepresentasikan pergerakan kamera horizontal
    psf_motion = np.zeros((7, 7), dtype=np.float32)
    psf_motion[3, 0:7] = 1/7 # Gerakan 7 piksel horizontal

    # 2. Terapkan blur ke gambar bersih (konvolusi)
    img_blurred_float = convolve2d(img_clean.astype(np.float32), psf_motion, mode='same', boundary='wrap')
    img_blurred = np.clip(img_blurred_float, 0, 255).astype(np.uint8)

    # 3. Tambahkan Gaussian Noise ke gambar buram
    mean_noise = 0
    var_noise = 100 # Varian noise
    sigma_noise = var_noise**0.5
    gaussian_noise_arr = np.random.normal(mean_noise, sigma_noise, img_clean.shape)
    img_degraded = np.clip(img_blurred + gaussian_noise_arr, 0, 255).astype(np.uint8)

    # 4. Terapkan Wiener Filter
    # K (noise-to-signal ratio) adalah parameter kunci.
    # Semakin kecil K, semakin agresif deblurring, tapi bisa memperkuat noise.
    # Semakin besar K, semakin banyak smoothing, tapi deblurring kurang efektif.
    snr_val = 10 # Contoh SNR. K = 1/SNR.
    restored_img_wiener = wiener_filter_deconvolution(img_degraded, psf_motion, snr=snr_val)

    # Tampilkan gambar
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Gambar Asli (Bersih)')
    plt.imshow(img_clean, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Gambar Terdegradasi (Blur + Noise)')
    plt.imshow(img_degraded, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f'Restored dengan Wiener Filter (SNR={snr_val})')
    plt.imshow(restored_img_wiener, cmap='gray')
    plt.axis('off')
    plt.show()
