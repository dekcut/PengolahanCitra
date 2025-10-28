import cv2
import numpy as np
import matplotlib.pyplot as plt

# Muat gambar grayscale
img_path = 'objek_bentuk.jpeg' # Gambar objek tunggal dengan latar belakang putih
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {img_path}")
else:
    # Binerisasi gambar (asumsikan objek gelap, latar belakang terang)
    # Jika objek terang, gunakan cv2.THRESH_BINARY_INV
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Temukan kontur objek
    # cv2.RETR_EXTERNAL untuk kontur terluar saja
    # cv2.CHAIN_APPROX_SIMPLE untuk menyimpan titik ujung saja
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Ambil kontur terbesar (asumsi itu objek utama)
        cnt = max(contours, key=cv2.contourArea)

        # --- Ekstraksi Fitur Geometris ---
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True) # True untuk kontur tertutup

        # Bounding Box Persegi Panjang (min_x, min_y, width, height)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0

        # Bounding Box Rotasi (center_x, center_y), (width, height), angle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box) # Ubah ke integer

        # Convex Hull
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0

        # Ekstent
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0

        # Momen
        M = cv2.moments(cnt)
        # cx = int(M['m10']/M['m00']) if M['m00'] != 0 else 0
        # cy = int(M['m01']/M['m00']) if M['m00'] != 0 else 0

        print(f"--- Fitur Geometris Objek ---")
        print(f"Luas (Area): {area:.2f} piksel")
        print(f"Perimeter: {perimeter:.2f} piksel")
        print(f"Rasio Aspek: {aspect_ratio:.2f}")
        print(f"Soliditas: {solidity:.2f}")
        print(f"Ekstent: {extent:.2f}")

        # Visualisasi
        img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Untuk menggambar warna
        cv2.drawContours(img_display, [cnt], 0, (0, 255, 0), 2) # Kontur asli (hijau)
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (255, 0, 0), 2) # Bounding box (biru)
        cv2.drawContours(img_display, [box], 0, (0, 0, 255), 2) # Bounding box rotasi (merah)
        cv2.drawContours(img_display, [hull], 0, (255, 255, 0), 2) # Convex hull (kuning)


        plt.figure(figsize=(8, 8))
        plt.title('Objek dengan Fitur Geometris')
        plt.imshow(img_display)
        plt.axis('off')
        plt.show()

    else:
        print("Tidak ada kontur yang ditemukan.")