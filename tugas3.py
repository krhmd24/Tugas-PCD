import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("hasil", exist_ok=True)

def clahe_enhancement(gray):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    hasil = clahe.apply(gray)

    return hasil

def gamma_correction(gray, gamma):

    tabel = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")


    hasil = cv2.LUT(gray, tabel)

    return hasil

def analisis_citra(gray):

    # Rata-rata intensitas
    rata_rata = np.mean(gray)

    # Standar deviasi sebagai ukuran kontras
    kontras = np.std(gray)

    # Persentase pixel gelap
    pixel_gelap = np.sum(gray < 50) / gray.size * 100

    # Persentase pixel terang
    pixel_terang = np.sum(gray > 220) / gray.size * 100


    # Menentukan kondisi citra

    if rata_rata < 90:

        kondisi = "Terlalu Gelap"
        metode = "Gamma Correction"

    elif rata_rata > 190:

        kondisi = "Terlalu Terang"
        metode = "Gamma Correction"

    elif kontras < 35:

        kondisi = "Contrast Rendah"
        metode = "CLAHE"

    else:

        kondisi = "Citra Cukup Baik"
        metode = "Tidak Memerlukan Enhancement"


    return (
        rata_rata,
        kontras,
        pixel_gelap,
        pixel_terang,
        kondisi,
        metode
    )


# ==========================================
# MEMPROSES SEMUA GAMBAR
# ==========================================

for i in range(1, 10):

    print("\n====================================")
    print(f"ANALISIS GAMBAR {i}")
    print("====================================")


    # ------------------------------------------
    # MEMBACA GAMBAR
    # ------------------------------------------

    nama_file = f"gambar/gambar_{i}.jpg"

    gambar = cv2.imread(nama_file)


    if gambar is None:

        print("Gambar tidak ditemukan!")

        continue


    # ------------------------------------------
    # BGR KE RGB
    # ------------------------------------------

    gambar_rgb = cv2.cvtColor(
        gambar,
        cv2.COLOR_BGR2RGB
    )


    # ------------------------------------------
    # BGR KE GRAYSCALE
    # ------------------------------------------

    gambar_gray = cv2.cvtColor(
        gambar,
        cv2.COLOR_BGR2GRAY
    )


    # ------------------------------------------
    # ANALISIS SEBELUM ENHANCEMENT
    # ------------------------------------------

    (
        rata_rata,
        kontras,
        pixel_gelap,
        pixel_terang,
        kondisi,
        metode

    ) = analisis_citra(gambar_gray)


    # ------------------------------------------
    # MELAKUKAN ENHANCEMENT
    # ------------------------------------------

    if kondisi == "Contrast Rendah":

        gambar_enhancement = clahe_enhancement(
            gambar_gray
        )


    elif kondisi == "Terlalu Terang":

        gambar_enhancement = gamma_correction(
            gambar_gray,
            2.0
        )


    elif kondisi == "Terlalu Gelap":

        gambar_enhancement = gamma_correction(
            gambar_gray,
            0.5
        )


    else:

        gambar_enhancement = gambar_gray


    # ------------------------------------------
    # ANALISIS SETELAH ENHANCEMENT
    # ------------------------------------------

    rata_rata_baru = np.mean(
        gambar_enhancement
    )

    kontras_baru = np.std(
        gambar_enhancement
    )


    # ------------------------------------------
    # MENAMPILKAN DATA
    # ------------------------------------------

    print("\nSEBELUM ENHANCEMENT")

    print(
        f"Rata-rata Intensitas : {rata_rata:.2f}"
    )

    print(
        f"Nilai Kontras        : {kontras:.2f}"
    )

    print(
        f"Pixel Gelap          : {pixel_gelap:.2f}%"
    )

    print(
        f"Pixel Terang         : {pixel_terang:.2f}%"
    )

    print(
        f"Kondisi Citra        : {kondisi}"
    )

    print(
        f"Metode Enhancement   : {metode}"
    )


    print("\nSETELAH ENHANCEMENT")

    print(
        f"Rata-rata Intensitas : {rata_rata_baru:.2f}"
    )

    print(
        f"Nilai Kontras        : {kontras_baru:.2f}"
    )


    # ==========================================
    # MEMBUAT TAMPILAN
    # ==========================================

    plt.figure(figsize=(16, 10))


    # ------------------------------------------
    # CITRA ASLI RGB
    # ------------------------------------------

    plt.subplot(2, 3, 1)

    plt.imshow(gambar_rgb)

    plt.title("Citra Asli RGB")

    plt.axis("off")


    # ------------------------------------------
    # CITRA GRAYSCALE
    # ------------------------------------------

    plt.subplot(2, 3, 2)

    plt.imshow(
        gambar_gray,
        cmap="gray"
    )

    plt.title("Citra Grayscale")

    plt.axis("off")


    # ------------------------------------------
    # HASIL ENHANCEMENT
    # ------------------------------------------

    plt.subplot(2, 3, 3)

    plt.imshow(
        gambar_enhancement,
        cmap="gray"
    )

    plt.title(
        f"Hasil Enhancement\n{metode}"
    )

    plt.axis("off")


    # ------------------------------------------
    # HISTOGRAM RGB
    # ------------------------------------------

    plt.subplot(2, 3, 4)


    plt.hist(
        gambar_rgb[:, :, 0].ravel(),
        bins=256,
        range=[0, 256],
        alpha=0.5,
        label="Red"
    )


    plt.hist(
        gambar_rgb[:, :, 1].ravel(),
        bins=256,
        range=[0, 256],
        alpha=0.5,
        label="Green"
    )


    plt.hist(
        gambar_rgb[:, :, 2].ravel(),
        bins=256,
        range=[0, 256],
        alpha=0.5,
        label="Blue"
    )


    plt.title("Histogram RGB")

    plt.xlabel("Intensitas Pixel")

    plt.ylabel("Jumlah Pixel")

    plt.legend()


    # ------------------------------------------
    # HISTOGRAM GRAYSCALE
    # ------------------------------------------

    plt.subplot(2, 3, 5)


    plt.hist(
        gambar_gray.ravel(),
        bins=256,
        range=[0, 256]
    )


    plt.title("Histogram Grayscale")

    plt.xlabel("Intensitas Pixel")

    plt.ylabel("Jumlah Pixel")


    # ------------------------------------------
    # HISTOGRAM SETELAH ENHANCEMENT
    # ------------------------------------------

    plt.subplot(2, 3, 6)


    plt.hist(
        gambar_enhancement.ravel(),
        bins=256,
        range=[0, 256]
    )


    plt.title(
        "Histogram Setelah Enhancement"
    )

    plt.xlabel("Intensitas Pixel")

    plt.ylabel("Jumlah Pixel")


    # ------------------------------------------
    # JUDUL UTAMA
    # ------------------------------------------

    plt.suptitle(

        f"ANALISIS CITRA IJAZAH - GAMBAR {i}\n"
        f"Kondisi: {kondisi} | "
        f"Metode: {metode}",

        fontsize=16
    )


    plt.tight_layout()


    # ------------------------------------------
    # MENYIMPAN HASIL
    # ------------------------------------------

    plt.savefig(
        f"hasil/hasil_gambar_{i}.png"
    )


    # Menampilkan
    plt.show()


print("\n====================================")
print("SEMUA GAMBAR SELESAI DIPROSES")
print("====================================")