import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def inverse_filter_deconvolution(img_degraded, psf, threshold=None):
    """
    Implementasi Inverse Filter atau Pseudo-Inverse Filter sederhana.
    
    Args:
        img_degraded (np.array): Gambar terdegradasi (grayscale).
        psf (np.array): Point Spread Function (kernel blur).
        threshold (float): Ambang batas untuk Pseudo-Inverse. Jika None, Inverse Filter murni.
                           Jika |H(u,v)| < threshold, set H(u,v) ke 1 (atau lain) untuk menghindari divisi nol.
    Returns:
        np.array: Gambar yang direstorasi.
    """
    img_float = img_degraded.astype(np.float64) / 255.0
    psf_float = psf.astype(np.float64) / psf.sum()

    # Transformasi Fourier
    G = np.fft.fft2(img_float)
    H = np.fft.fft2(psf_float, s=img_float.shape)

    # Inisialisasi filter
    deconv_filter = np.zeros_like(H, dtype=np.complex128)

    if threshold is None: # Inverse Filter murni
        # Handle division by zero or very small values in H
        epsilon = 1e-10 # Small constant to avoid division by zero
        deconv_filter = 1 / (H + epsilon)
    else: # Pseudo-Inverse Filter
        # Avoid division by small values of H by setting them to 1 (or other value)
        # More correctly: set the corresponding F_hat to 0
        idx = np.abs(H) > threshold
        deconv_filter[idx] = 1 / H[idx]

    F_hat = deconv_filter * G
    restored_img = np.fft.ifft2(F_hat)
    restored_img = np.abs(restored_img)
    restored_img = np.clip(restored_img * 255, 0, 255).astype(np.uint8)

    return restored_img

# --- Simulasi Blur (Motion Blur) dan Tambahkan Sedikit Noise ---
img_path = 'gambar_bersih.jpeg'
img_clean = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img_clean is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Buat PSF untuk motion blur
    psf_motion = np.zeros((7, 7), dtype=np.float32)
    psf_motion[3, 0:7] = 1/7

    # Terapkan blur
    img_blurred_float = convolve2d(img_clean.astype(np.float32), psf_motion, mode='same', boundary='wrap')
    img_blurred = np.clip(img_blurred_float, 0, 255).astype(np.uint8)

    # Tambahkan sedikit Gaussian Noise (penting untuk melihat efek inverse filter)
    mean_noise = 0
    var_noise = 10 # Varian noise sangat kecil
    sigma_noise = var_noise**0.5
    gaussian_noise_arr = np.random.normal(mean_noise, sigma_noise, img_clean.shape)
    img_degraded_noisy = np.clip(img_blurred + gaussian_noise_arr, 0, 255).astype(np.uint8)

    # Terapkan Inverse Filter (akan sangat noise)
    restored_inverse = inverse_filter_deconvolution(img_degraded_noisy, psf_motion, threshold=None)

    # Terapkan Pseudo-Inverse Filter (dengan threshold)
    # Threshold yang tepat sangat penting. Umumnya, dipilih berdasarkan energi noise atau H.
    # Di sini, kita coba nilai heuristik.
    restored_pseudo_inverse = inverse_filter_deconvolution(img_degraded_noisy, psf_motion, threshold=0.01)

    # Tampilkan gambar
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Gambar Terdegradasi (Blur + Sedikit Noise)')
    plt.imshow(img_degraded_noisy, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Restored dengan Inverse Filter (Noisy!)')
    plt.imshow(restored_inverse, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Restored dengan Pseudo-Inverse Filter (Threshold=0.01)')
    plt.imshow(restored_pseudo_inverse, cmap='gray')
    plt.axis('off')
    plt.show()
