import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(img, seed_point, threshold_diff):
    """
    Implementasi Region Growing sederhana.
    Args:
        img (np.array): Gambar grayscale.
        seed_point (tuple): Koordinat (row, col) dari seed point.
        threshold_diff (int): Batas perbedaan intensitas untuk menambahkan piksel.
    Returns:
        np.array: Gambar biner hasil segmentasi.
    """
    if img is None:
        return None

    h, w = img.shape
    segmented = np.zeros_like(img, dtype=np.uint8)
    
    # Gunakan antrian untuk piksel yang akan diperiksa
    queue = []
    # Set seed point
    queue.append(seed_point)
    segmented[seed_point] = 255 # Tandai sebagai bagian dari segmen
    
    # Nilai intensitas seed
    seed_intensity = img[seed_point]

    # Iterasi selama antrian tidak kosong
    while len(queue) > 0:
        current_pixel = queue.pop(0)
        r, c = current_pixel

        # Periksa 8 tetangga
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue # Skip current pixel itself

                nr, nc = r + dr, c + dc # Neighbor row, col

                # Cek batas gambar
                if 0 <= nr < h and 0 <= nc < w:
                    # Jika belum disegmentasi dan memenuhi kriteria kesamaan
                    if segmented[nr, nc] == 0 and abs(int(img[nr, nc]) - int(seed_intensity)) <= threshold_diff:
                        segmented[nr, nc] = 255
                        queue.append((nr, nc))
    
    return segmented


# Muat gambar grayscale
img_path = 'objek_homogen.jpeg' # Ganti dengan path gambar Anda
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Pilih seed point (misalnya, di tengah objek)
    seed_point = (img.shape[0] // 2, img.shape[1] // 2)
    # Tentukan threshold perbedaan intensitas
    threshold_diff = 15 # Piksel dengan beda intensitas <= 15 akan ditambahkan

    # Lakukan Region Growing
    segmented_img = region_growing(img, seed_point, threshold_diff)

    # Tampilkan gambar
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Gambar Asli dengan Seed Point')
    plt.imshow(img, cmap='gray')
    plt.plot(seed_point[1], seed_point[0], 'rx') # Tandai seed point
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Segmented (Region Growing, T_diff={threshold_diff})')
    plt.imshow(segmented_img, cmap='gray')
    plt.axis('off')
    plt.show()