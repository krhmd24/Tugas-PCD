
import os
import re
import cv2
import pytesseract


# ==========================================
# PENGATURAN PROGRAM
# ==========================================

# Lokasi aplikasi Tesseract OCR di Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Folder sumber gambar dan folder hasil
FOLDER_GAMBAR = "gambar"
FOLDER_OUTPUT = "hasil_ocr"

os.makedirs(FOLDER_OUTPUT, exist_ok=True)

# Daftar gambar yang akan diproses
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

# Nomor ijazah yang menjadi acuan
NOMOR_ACUAN = "571012022000056"

# Koordinat pemotongan area nomor ijazah
BATAS_CROP = (50, 2100, 1300, 2420)

# Konfigurasi Tesseract OCR
KONFIG_OCR = "--oem 3 --psm 6"


# ==========================================
# METODE ENHANCEMENT
# ==========================================

def tambah_kecerahan(citra, nilai=30):
    """Meningkatkan kecerahan gambar."""
    citra_baru = cv2.convertScaleAbs(
        citra,
        alpha=1.0,
        beta=nilai
    )
    return citra_baru


def perbaiki_kontras(citra):
    """Meningkatkan kontras dengan contrast stretching."""
    citra_float = citra.astype("float32")

    nilai_min = citra_float.min()
    nilai_max = citra_float.max()

    if nilai_min == nilai_max:
        return citra.copy()

    citra_baru = (
        255.0 * (citra_float - nilai_min)
        / (nilai_max - nilai_min)
    )

    return citra_baru.clip(0, 255).astype("uint8")


def seimbangkan_histogram(citra):
    """Menerapkan histogram equalization."""
    return cv2.equalizeHist(citra)


# ==========================================
# MEMBACA NOMOR DENGAN OCR
# ==========================================

def baca_nomor(citra):
    """Membaca teks menggunakan Tesseract dan mengambil angka."""
    teks = pytesseract.image_to_string(
        citra,
        config=KONFIG_OCR
    )

    nomor = re.sub(r"[^0-9]", "", teks)
    return nomor


# ==========================================
# MEMBACA DAN MEMOTONG GAMBAR
# ==========================================

def siapkan_area_nomor(lokasi_gambar):
    """Membaca gambar, memutarnya, mengubah grayscale, lalu crop."""

    gambar = cv2.imread(lokasi_gambar)

    if gambar is None:
        print(f"Gambar tidak dapat dibaca: {lokasi_gambar}")
        return None

    # Memutar gambar 90 derajat searah jarum jam
    gambar = cv2.rotate(
        gambar,
        cv2.ROTATE_90_CLOCKWISE
    )

    # Mengubah gambar menjadi grayscale
    grayscale = cv2.cvtColor(
        gambar,
        cv2.COLOR_BGR2GRAY
    )

    # Mengambil koordinat crop
    x_awal, y_awal, x_akhir, y_akhir = BATAS_CROP

    area_nomor = grayscale[
        y_awal:y_akhir,
        x_awal:x_akhir
    ]

    if area_nomor.size == 0:
        print(f"Area crop kosong: {lokasi_gambar}")
        print("Periksa kembali koordinat crop.")
        return None

    return area_nomor


# ==========================================
# MEMPROSES SEMUA METODE
# ==========================================

def proses_ocr_semua_metode(area_nomor):
    """Menjalankan OCR pada gambar asli dan hasil enhancement."""

    versi_citra = {
        "Asli": area_nomor,
        "Brightness Adjustment": tambah_kecerahan(area_nomor),
        "Contrast Stretching": perbaiki_kontras(area_nomor),
        "Histogram Equalization": seimbangkan_histogram(area_nomor),
    }

    hasil_baca = {}

    for nama_metode, citra in versi_citra.items():
        hasil_baca[nama_metode] = baca_nomor(citra)

    return versi_citra, hasil_baca


# ==========================================
# MENYIMPAN GAMBAR HASIL
# ==========================================

def simpan_citra(versi_citra, nomor_gambar):
    """Menyimpan gambar asli dan hasil enhancement."""

    nama_file_output = {
        "Asli": "asli",
        "Brightness Adjustment": "brightness",
        "Contrast Stretching": "contrast",
        "Histogram Equalization": "histogram",
    }

    for metode, citra in versi_citra.items():
        nama_output = (
            f"gambar_{nomor_gambar}_"
            f"{nama_file_output[metode]}.png"
        )

        lokasi_output = os.path.join(
            FOLDER_OUTPUT,
            nama_output
        )

        cv2.imwrite(lokasi_output, citra)


# ==========================================
# MENAMPILKAN DAN MENYUSUN LAPORAN OCR
# ==========================================

def cetak_dan_buat_laporan(
    nomor_gambar,
    nama_file,
    hasil_baca
):
    """Menampilkan hasil OCR dan membuat catatan laporan."""

    status = {}

    for metode, hasil in hasil_baca.items():
        if metode != "Asli":
            status[metode] = (
                "COCOK"
                if hasil == NOMOR_ACUAN
                else "TIDAK COCOK"
            )

    print("\n" + "=" * 45)
    print(f"HASIL OCR GAMBAR {nomor_gambar}")
    print(f"File: {nama_file}")
    print("=" * 45)

    print("\nHASIL PEMBACAAN:")

    for metode, hasil in hasil_baca.items():
        print(f"{metode:<25}: {hasil}")

    print("\nPERBANDINGAN DENGAN NOMOR ACUAN")
    print(f"Nomor target: {NOMOR_ACUAN}")

    for metode, hasil_status in status.items():
        print(f"{metode:<25}: {hasil_status}")

    laporan = [
        f"GAMBAR {nomor_gambar}",
        f"File: {nama_file}",
        "",
        f"Nomor target: {NOMOR_ACUAN}",
        "",
        f"Asli: {hasil_baca['Asli']}",
        "",
    ]

    for metode in (
        "Brightness Adjustment",
        "Contrast Stretching",
        "Histogram Equalization"
    ):
        laporan.extend([
            f"{metode}:",
            hasil_baca[metode],
            f"Status: {status[metode]}",
            "",
        ])

    laporan.append("-" * 45)

    return "\n".join(laporan)


# ==========================================
# PROGRAM UTAMA
# ==========================================

def main():
    semua_laporan = []
    jumlah_berhasil_dibaca = 0

    print("=" * 45)
    print("PROGRAM OCR NOMOR IJAZAH")
    print("=" * 45)

    for urutan, nama_file in enumerate(
        DAFTAR_GAMBAR,
        start=1
    ):
        lokasi_gambar = os.path.join(
            FOLDER_GAMBAR,
            nama_file
        )

        print(f"\nMemproses gambar: {nama_file}")

        area_nomor = siapkan_area_nomor(lokasi_gambar)

        if area_nomor is None:
            continue

        versi_citra, hasil_baca = proses_ocr_semua_metode(
            area_nomor
        )

        simpan_citra(versi_citra, urutan)

        catatan = cetak_dan_buat_laporan(
            urutan,
            nama_file,
            hasil_baca
        )

        semua_laporan.append(catatan)
        jumlah_berhasil_dibaca += 1

    # Menyimpan seluruh hasil OCR ke file teks
    lokasi_laporan = os.path.join(
        FOLDER_OUTPUT,
        "hasil_ocr.txt"
    )

    with open(
        lokasi_laporan,
        "w",
        encoding="utf-8"
    ) as file_laporan:
        file_laporan.write("\n\n".join(semua_laporan))

    print("\n" + "=" * 45)
    print("PEMROSESAN OCR SELESAI")
    print(f"Jumlah gambar berhasil diproses: {jumlah_berhasil_dibaca}")
    print(f"Folder hasil: {FOLDER_OUTPUT}")
    print(f"Laporan OCR: {lokasi_laporan}")
    print("=" * 45)


if __name__ == "__main__":
    main()
