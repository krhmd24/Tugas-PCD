import os
import re
import cv2
import pytesseract


# Pengaturan Tesseract OCR pada Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

FOLDER_OUTPUT = "hasil_ocr"
os.makedirs(FOLDER_OUTPUT, exist_ok=True)

FILE_GAMBAR = [
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

NOMOR_ACUAN = "571012022000056"
BATAS_CROP = (50, 2100, 1300, 2420)
KONFIG_OCR = "--oem 3 --psm 6"


def tambah_kecerahan(citra, nilai=30):
    """Menaikkan kecerahan citra."""
    return cv2.convertScaleAbs(citra, alpha=1.0, beta=nilai)


def perbaiki_kontras(citra):
    """Memetakan rentang intensitas citra ke 0-255."""
    data = citra.astype("float32")
    batas_bawah = data.min()
    batas_atas = data.max()

    if batas_bawah == batas_atas:
        return citra.copy()

    hasil = 255.0 * (data - batas_bawah) / (batas_atas - batas_bawah)
    return hasil.clip(0, 255).astype("uint8")


def seimbangkan_histogram(citra):
    """Menerapkan histogram equalization."""
    return cv2.equalizeHist(citra)


def baca_nomor(citra):
    """Membaca teks pada citra menggunakan Tesseract, lalu mengambil angka."""
    teks_terbaca = pytesseract.image_to_string(
        citra,
        config=KONFIG_OCR
    )
    return re.sub(r"[^0-9]", "", teks_terbaca.upper())


def siapkan_area_nomor(nama_file):
    """Baca gambar, putar, ubah ke grayscale, dan potong area nomor."""
    gambar = cv2.imread(nama_file)

    if gambar is None:
        print("Gambar tidak ditemukan!")
        return None

    gambar = cv2.rotate(gambar, cv2.ROTATE_90_CLOCKWISE)
    grayscale = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    x_awal, y_awal, x_akhir, y_akhir = BATAS_CROP
    area_nomor = grayscale[y_awal:y_akhir, x_awal:x_akhir]

    if area_nomor.size == 0:
        print("Crop gagal!")
        return None

    return area_nomor


def proses_ocr_semua_metode(area_nomor):
    """Jalankan OCR pada citra asli dan ketiga hasil enhancement."""
    versi_citra = {
        "Asli": area_nomor,
        "Brightness Adjustment": tambah_kecerahan(area_nomor),
        "Contrast Stretching": perbaiki_kontras(area_nomor),
        "Histogram Equalization": seimbangkan_histogram(area_nomor),
    }

    # OCR dijalankan untuk setiap versi citra
    hasil_baca = {
        nama_metode: baca_nomor(citra)
        for nama_metode, citra in versi_citra.items()
    }

    return versi_citra, hasil_baca


def simpan_citra(versi_citra, nomor_gambar):
    """Simpan citra asli dan hasil enhancement."""
    nama_output = {
        "Asli": "asli",
        "Brightness Adjustment": "brightness",
        "Contrast Stretching": "contrast",
        "Histogram Equalization": "histogram",
    }

    for metode, citra in versi_citra.items():
        alamat = os.path.join(
            FOLDER_OUTPUT,
            f"gambar_{nomor_gambar}_{nama_output[metode]}.png"
        )
        cv2.imwrite(alamat, citra)


def cetak_dan_buat_laporan(nomor_gambar, nama_file, hasil_baca):
    """Tampilkan kecocokan OCR dan susun catatan untuk file laporan."""
    status = {
        metode: ("COCOK" if hasil == NOMOR_ACUAN else "TIDAK COCOK")
        for metode, hasil in hasil_baca.items()
        if metode != "Asli"
    }

    print("\n====================================")
    print(f"OCR NOMOR IJAZAH - GAMBAR {nomor_gambar}")
    print(f"File : {nama_file}")
    print("====================================")
    print("\nHASIL OCR:")

    for metode, hasil in hasil_baca.items():
        print(f"{metode:<23}: {hasil}")

    print("\nPERBANDINGAN DENGAN NOMOR TARGET:")
    print(f"Target : {NOMOR_ACUAN}")

    for metode, hasil_status in status.items():
        print(f"{metode:<23}: {hasil_status}")

    bagian_laporan = [
        f"GAMBAR {nomor_gambar}",
        f"File: {nama_file}",
        "",
        f"Target:\n{NOMOR_ACUAN}",
        "",
        f"Asli:\n{hasil_baca['Asli']}",
        "",
    ]

    for metode in ("Brightness Adjustment", "Contrast Stretching",
                   "Histogram Equalization"):
        bagian_laporan.extend([
            f"{metode}:",
            hasil_baca[metode],
            f"Status: {status[metode]}",
            "",
        ])

    bagian_laporan.append("----------------------------------------")
    return "\n".join(bagian_laporan)


def main():
    semua_laporan = []

    for urutan, nama_file in enumerate(FILE_GAMBAR, start=1):
        area_nomor = siapkan_area_nomor(nama_file)

        if area_nomor is None:
            continue

        citra_versi, hasil_baca = proses_ocr_semua_metode(area_nomor)
        simpan_citra(citra_versi, urutan)

        catatan = cetak_dan_buat_laporan(
            urutan,
            nama_file,
            hasil_baca
        )
        semua_laporan.append(catatan)

    lokasi_laporan = os.path.join(FOLDER_OUTPUT, "hasil_ocr.txt")
    with open(lokasi_laporan, "w", encoding="utf-8") as file_laporan:
        file_laporan.write("\n\n".join(semua_laporan))

    print("\n====================================")
    print("OCR SEMUA GAMBAR SELESAI")
    print(f"Hasil disimpan di folder: {FOLDER_OUTPUT}")
    print("====================================")


if __name__ == "__main__":
    main()
