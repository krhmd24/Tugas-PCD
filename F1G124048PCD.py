import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Folder untuk menyimpan gambar hasil pemrosesan
FOLDER_HASIL = "hasil"
os.makedirs(FOLDER_HASIL, exist_ok=True)

# Nama-nama gambar yang akan dianalisis
DAFTAR_GAMBAR = [
    "01_HighQuality_Enhanced.jpg",
    "02_LowContrast.jpg",
    "03_Blurred.jpg",
    "04_HighNoise.jpg",
    "05_LowResolution_Upsampled.jpg",
    "06_Faded_Underexposed.jpg",
    "07_ColorShift_WarmTint.jpg",
    "08_JPEGCompression_Artifacts.jpg",
    "09_CombinedDegradation.jpg",
]

# Batas area nomor ijazah
AREA_CROP = (50, 2100, 1300, 2420)


def atur_kecerahan(citra, tambahan=30):
    """Menambah kecerahan citra dengan nilai tetap."""
    return cv2.convertScaleAbs(citra, alpha=1.0, beta=tambahan)


def rentangkan_kontras(citra):
    """Meregangkan intensitas piksel ke rentang 0 sampai 255."""
    citra_float = citra.astype(np.float32)
    nilai_min = np.min(citra_float)
    nilai_max = np.max(citra_float)

    # Hindari pembagian dengan nol jika semua piksel bernilai sama
    if nilai_min == nilai_max:
        return citra.copy()

    citra_hasil = (citra_float - nilai_min) * 255.0 / (nilai_max - nilai_min)
    citra_hasil = np.clip(citra_hasil, 0, 255)
    return citra_hasil.astype(np.uint8)


def ratakan_histogram(citra):
    """Menerapkan histogram equalization pada citra grayscale."""
    return cv2.equalizeHist(citra)


def hitung_statistik(citra):
    """Menghitung intensitas rata-rata, kontras, serta piksel gelap dan terang."""
    intensitas_rata = np.mean(citra)
    nilai_kontras = np.std(citra)
    persen_gelap = np.count_nonzero(citra < 50) / citra.size * 100
    persen_terang = np.count_nonzero(citra > 220) / citra.size * 100

    return {
        "rata_rata": intensitas_rata,
        "kontras": nilai_kontras,
        "pixel_gelap": persen_gelap,
        "pixel_terang": persen_terang,
    }


def tampilkan_statistik(judul, data, lengkap=False):
    print(f"\n{judul}")
    print(f"Rata-rata Intensitas : {data['rata_rata']:.2f}")
    print(f"Nilai Kontras        : {data['kontras']:.2f}")

    if lengkap:
        print(f"Pixel Gelap          : {data['pixel_gelap']:.2f}%")
        print(f"Pixel Terang         : {data['pixel_terang']:.2f}%")


def buat_perbandingan(citra_asli, hasil_metode, nomor, nama_file):
    """Menyusun gambar dan histogram untuk citra asli dan tiap metode."""
    daftar_panel = [
        ("Sebelum Enhancement", citra_asli),
        ("Brightness Adjustment", hasil_metode["Brightness Adjustment"]),
        ("Contrast Stretching", hasil_metode["Contrast Stretching"]),
        ("Histogram Equalization", hasil_metode["Histogram Equalization"]),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(15, 14))

    for baris, (nama_metode, citra) in enumerate(daftar_panel):
        # Kolom kiri: citra hasil
        axes[baris, 0].imshow(citra, cmap="gray")
        axes[baris, 0].set_title(nama_metode)
        axes[baris, 0].axis("off")

        # Kolom kanan: histogram intensitas piksel
        axes[baris, 1].hist(citra.ravel(), bins=256, range=[0, 256])

        if baris == 0:
            judul_histogram = "Histogram Sebelum Enhancement"
        elif baris == 1:
            judul_histogram = "Histogram Brightness Adjustment"
        elif baris == 2:
            judul_histogram = "Histogram Contrast Stretching"
        else:
            judul_histogram = "Histogram Setelah Equalization"

        axes[baris, 1].set_title(judul_histogram)
        axes[baris, 1].set_xlabel("Intensitas Pixel")
        axes[baris, 1].set_ylabel("Jumlah Pixel")

    fig.suptitle(
        f"ENHANCEMENT NOMOR IJAZAH - GAMBAR {nomor}\n{nama_file}",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    lokasi_simpan = os.path.join(FOLDER_HASIL, f"hasil_nomor_ijazah_{nomor}.png")
    plt.savefig(lokasi_simpan, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def proses_gambar(nomor, nama_file):
    print("\n====================================")
    print(f"ANALISIS NOMOR IJAZAH - GAMBAR {nomor}")
    print(f"File : {nama_file}")
    print("====================================")

    citra = cv2.imread(nama_file)
    if citra is None:
        print("Gambar tidak ditemukan!")
        return

    # Samakan orientasi gambar, lalu ubah ke grayscale
    citra = cv2.rotate(citra, cv2.ROTATE_90_CLOCKWISE)
    grayscale = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)

    # Ambil area nomor ijazah berdasarkan koordinat yang ditentukan
    x1, y1, x2, y2 = AREA_CROP
    potongan = grayscale[y1:y2, x1:x2]

    if potongan.size == 0:
        print("Crop nomor ijazah gagal!")
        return

    # Terapkan tiga teknik enhancement pada potongan yang sama
    hasil_metode = {
        "Brightness Adjustment": atur_kecerahan(potongan),
        "Contrast Stretching": rentangkan_kontras(potongan),
        "Histogram Equalization": ratakan_histogram(potongan),
    }

    # Analisis citra asli dan seluruh hasil enhancement
    statistik_asli = hitung_statistik(potongan)
    statistik_hasil = {
        nama: hitung_statistik(citra_hasil)
        for nama, citra_hasil in hasil_metode.items()
    }

    tampilkan_statistik("SEBELUM ENHANCEMENT", statistik_asli, lengkap=True)
    tampilkan_statistik(
        "BRIGHTNESS ADJUSTMENT",
        statistik_hasil["Brightness Adjustment"],
    )
    tampilkan_statistik(
        "CONTRAST STRETCHING",
        statistik_hasil["Contrast Stretching"],
    )
    tampilkan_statistik(
        "HISTOGRAM EQUALIZATION",
        statistik_hasil["Histogram Equalization"],
    )

    # Tentukan metode dengan nilai standar deviasi tertinggi
    perbandingan_kontras = {
        nama: data["kontras"] for nama, data in statistik_hasil.items()
    }
    metode_tertinggi = max(perbandingan_kontras, key=perbandingan_kontras.get)
    print(
        "\nKontras tertinggi berdasarkan "
        f"standar deviasi : {metode_tertinggi}"
    )

    buat_perbandingan(potongan, hasil_metode, nomor, nama_file)


for nomor, nama_file in enumerate(DAFTAR_GAMBAR, start=1):
    proses_gambar(nomor, nama_file)

print("\n====================================")
print("SEMUA GAMBAR SELESAI DIPROSES")
print("====================================")
