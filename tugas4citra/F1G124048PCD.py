
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# PENGATURAN PROGRAM
# ==========================================

FOLDER_GAMBAR = "gambar"
FOLDER_HASIL = "hasil"

os.makedirs(FOLDER_HASIL, exist_ok=True)

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


# ==========================================
# FUNGSI ENHANCEMENT
# ==========================================

def brightness_adjustment(citra, nilai=30):
    """Mengatur kecerahan citra."""
    return cv2.convertScaleAbs(
        citra,
        alpha=1.0,
        beta=nilai
    )


def contrast_stretching(citra):
    """Memperluas rentang intensitas piksel."""
    nilai_min = np.min(citra)
    nilai_max = np.max(citra)

    if nilai_min == nilai_max:
        return citra.copy()

    hasil = (
        (citra.astype(np.float32) - nilai_min)
        * 255.0
        / (nilai_max - nilai_min)
    )

    return np.clip(hasil, 0, 255).astype(np.uint8)


def histogram_equalization(citra):
    """Meningkatkan kontras melalui pemerataan histogram."""
    return cv2.equalizeHist(citra)


# ==========================================
# ANALISIS STATISTIK CITRA
# ==========================================

def analisis_citra(citra):
    """Menghitung rata-rata, kontras, piksel gelap, dan terang."""

    rata_rata = np.mean(citra)
    kontras = np.std(citra)

    total_piksel = citra.size

    piksel_gelap = (
        np.sum(citra < 50) / total_piksel
    ) * 100

    piksel_terang = (
        np.sum(citra > 200) / total_piksel
    ) * 100

    return {
        "rata_rata": rata_rata,
        "kontras": kontras,
        "piksel_gelap": piksel_gelap,
        "piksel_terang": piksel_terang,
    }


# ==========================================
# MENAMPILKAN STATISTIK
# ==========================================

def tampilkan_statistik(nama, citra):
    statistik = analisis_citra(citra)

    print(f"\n{nama}")
    print("-" * 40)
    print(
        f"Rata-rata Intensitas : "
        f"{statistik['rata_rata']:.2f}"
    )
    print(
        f"Nilai Kontras        : "
        f"{statistik['kontras']:.2f}"
    )
    print(
        f"Pixel Gelap          : "
        f"{statistik['piksel_gelap']:.2f}%"
    )
    print(
        f"Pixel Terang         : "
        f"{statistik['piksel_terang']:.2f}%"
    )

    return statistik


# ==========================================
# MENAMPILKAN DAN MENYIMPAN HISTOGRAM
# ==========================================

def simpan_perbandingan(
    citra_asli,
    hasil_metode,
    nama_file
):
    """Menyimpan perbandingan citra dan histogram."""

    metode = {
        "Asli": citra_asli,
        "Brightness Adjustment": hasil_metode[
            "Brightness Adjustment"
        ],
        "Contrast Stretching": hasil_metode[
            "Contrast Stretching"
        ],
        "Histogram Equalization": hasil_metode[
            "Histogram Equalization"
        ],
    }

    fig, axes = plt.subplots(
        4, 2,
        figsize=(12, 16)
    )

    for baris, (nama, citra) in enumerate(metode.items()):
        # Tampilan citra
        axes[baris, 0].imshow(
            citra,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[baris, 0].set_title(nama)
        axes[baris, 0].axis("off")

        # Histogram
        axes[baris, 1].hist(
            citra.ravel(),
            bins=256,
            range=(0, 256)
        )
        axes[baris, 1].set_title(
            f"Histogram - {nama}"
        )
        axes[baris, 1].set_xlim(0, 255)
        axes[baris, 1].set_xlabel(
            "Intensitas Piksel"
        )
        axes[baris, 1].set_ylabel(
            "Jumlah Piksel"
        )

    plt.tight_layout()

    nama_output = os.path.splitext(nama_file)[0]
    lokasi_plot = os.path.join(
        FOLDER_HASIL,
        f"{nama_output}_perbandingan.png"
    )

    plt.savefig(
        lokasi_plot,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# MENYIMPAN CITRA HASIL ENHANCEMENT
# ==========================================

def simpan_hasil(citra, nama_file, metode):
    nama_dasar = os.path.splitext(nama_file)[0]

    nama_metode = (
        metode.lower()
        .replace(" ", "_")
    )

    lokasi = os.path.join(
        FOLDER_HASIL,
        f"{nama_dasar}_{nama_metode}.jpg"
    )

    cv2.imwrite(lokasi, citra)


# ==========================================
# MEMPROSES SATU CITRA
# ==========================================

def proses_gambar(nama_file):
    lokasi_gambar = os.path.join(
        FOLDER_GAMBAR,
        nama_file
    )

    citra = cv2.imread(
        lokasi_gambar,
        cv2.IMREAD_GRAYSCALE
    )

    if citra is None:
        print(
            f"\nGagal membaca gambar: {lokasi_gambar}"
        )
        return

    print("\n" + "=" * 55)
    print(f"ANALISIS GAMBAR: {nama_file}")
    print("=" * 55)

    tampilkan_statistik(
        "SEBELUM ENHANCEMENT",
        citra
    )

    hasil_metode = {
        "Brightness Adjustment": brightness_adjustment(
            citra
        ),
        "Contrast Stretching": contrast_stretching(
            citra
        ),
        "Histogram Equalization": histogram_equalization(
            citra
        ),
    }

    for nama_metode, citra_hasil in hasil_metode.items():
        tampilkan_statistik(
            f"SETELAH {nama_metode.upper()}",
            citra_hasil
        )

        simpan_hasil(
            citra_hasil,
            nama_file,
            nama_metode
        )

    simpan_perbandingan(
        citra,
        hasil_metode,
        nama_file
    )

    print("\nHasil citra dan histogram berhasil disimpan.")


# ==========================================
# PROGRAM UTAMA
# ==========================================

def main():
    print("=" * 55)
    print("PROGRAM ENHANCEMENT CITRA NOMOR IJAZAH")
    print("=" * 55)

    jumlah_berhasil = 0

    for nama_file in DAFTAR_GAMBAR:
        lokasi = os.path.join(
            FOLDER_GAMBAR,
            nama_file
        )

        if not os.path.isfile(lokasi):
            print(f"\nFile tidak ditemukan: {lokasi}")
            continue

        proses_gambar(nama_file)
        jumlah_berhasil += 1

    print("\n" + "=" * 55)
    print("PEMROSESAN SELESAI")
    print(
        f"Jumlah gambar berhasil diproses: "
        f"{jumlah_berhasil}/{len(DAFTAR_GAMBAR)}"
    )
    print(f"Folder hasil: {FOLDER_HASIL}")
    print("=" * 55)


if __name__ == "__main__":
    main()
