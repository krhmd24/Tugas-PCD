
import os
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
from pathlib import Path


# ============================================================
# KONFIGURASI
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

FOLDER_INPUT = Path("gambar")
FOLDER_OUTPUT = Path("hasil")

NOMOR_TARGET = "571012022000056"

# Koordinat crop setelah gambar diputar
KOORDINAT_CROP = (540, 2240, 1150, 2340)

UKURAN_FILTER = 5

KONFIG_OCR = (
    "--oem 3 --psm 8 "
    "-c tessedit_char_whitelist=0123456789"
)

FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNGSI PENGOLAHAN TEKS OCR
# ============================================================

def ambil_angka(teks):
    """Menghapus karakter selain angka dari hasil OCR."""
    return re.sub(r"[^0-9]", "", teks)


def hitung_jarak(teks_target, teks_ocr):
    """Menghitung perbedaan karakter menggunakan Levenshtein."""
    baris = list(range(len(teks_ocr) + 1))

    for i, karakter_target in enumerate(teks_target, start=1):
        nilai_sebelumnya = baris[0]
        baris[0] = i

        for j, karakter_ocr in enumerate(teks_ocr, start=1):
            nilai_lama = baris[j]

            biaya = (
                0 if karakter_target == karakter_ocr else 1
            )

            baris[j] = min(
                baris[j] + 1,
                baris[j - 1] + 1,
                nilai_sebelumnya + biaya
            )

            nilai_sebelumnya = nilai_lama

    return baris[-1]


def evaluasi_ocr(teks_ocr):
    """Menghitung jumlah karakter benar dan persentase akurasi."""
    jarak = hitung_jarak(NOMOR_TARGET, teks_ocr)

    karakter_benar = max(0, len(NOMOR_TARGET) - jarak)

    akurasi = (
        karakter_benar / len(NOMOR_TARGET)
    ) * 100

    return karakter_benar, akurasi


# ============================================================
# MEMBACA, MEMUTAR, DAN MEMOTONG CITRA
# ============================================================

def ambil_area_nomor(lokasi):
    """Mengambil area nomor ijazah dalam bentuk grayscale."""

    citra = cv2.imread(str(lokasi))

    if citra is None:
        print(f"Gagal membaca gambar: {lokasi}")
        return None

    citra = cv2.rotate(
        citra,
        cv2.ROTATE_90_CLOCKWISE
    )

    x1, y1, x2, y2 = KOORDINAT_CROP

    tinggi, lebar = citra.shape[:2]

    if x1 < 0 or y1 < 0 or x2 > lebar or y2 > tinggi:
        print(f"Koordinat crop berada di luar gambar: {lokasi}")
        return None

    area = citra[y1:y2, x1:x2]

    if area.size == 0:
        print(f"Area nomor kosong: {lokasi}")
        return None

    grayscale = cv2.cvtColor(
        area,
        cv2.COLOR_BGR2GRAY
    )

    return grayscale


# ============================================================
# METODE FILTERING DAN SHARPENING
# ============================================================

def buat_variasi_citra(citra):
    """Menghasilkan citra dengan lima metode pemrosesan."""

    kernel_tajam = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    variasi = {
        "Original": citra,

        "Mean Filter": cv2.blur(
            citra,
            (UKURAN_FILTER, UKURAN_FILTER)
        ),

        "Median Filter": cv2.medianBlur(
            citra,
            UKURAN_FILTER
        ),

        "Gaussian Filter": cv2.GaussianBlur(
            citra,
            (UKURAN_FILTER, UKURAN_FILTER),
            0
        ),

        "Sharpening": cv2.filter2D(
            citra,
            -1,
            kernel_tajam
        )
    }

    return variasi


# ============================================================
# MEMBACA NOMOR MENGGUNAKAN TESSERACT
# ============================================================

def jalankan_ocr(citra):
    """Memperbesar citra sebelum melakukan pembacaan OCR."""

    citra_besar = cv2.resize(
        citra,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    teks = pytesseract.image_to_string(
        citra_besar,
        config=KONFIG_OCR
    )

    nomor = ambil_angka(teks)

    return nomor, citra_besar


# ============================================================
# MENYIMPAN HASIL GAMBAR
# ============================================================

def simpan_citra(citra, folder, nama_metode):
    """Menyimpan citra hasil pemrosesan."""

    nama_file = (
        nama_metode.lower().replace(" ", "_") + ".jpg"
    )

    lokasi_simpan = folder / nama_file

    cv2.imwrite(str(lokasi_simpan), citra)


# ============================================================
# MEMPROSES SATU GAMBAR
# ============================================================

def proses_satu_gambar(file_gambar):
    """Menjalankan semua metode dan mengevaluasi hasil OCR."""

    print("\n" + "=" * 70)
    print(f"NAMA GAMBAR: {file_gambar.name}")
    print("=" * 70)

    citra_asli = ambil_area_nomor(file_gambar)

    if citra_asli is None:
        return []

    kumpulan_citra = buat_variasi_citra(citra_asli)

    folder_gambar = FOLDER_OUTPUT / file_gambar.stem
    folder_gambar.mkdir(parents=True, exist_ok=True)

    data_gambar = []

    for metode, citra_hasil in kumpulan_citra.items():

        nomor_ocr, citra_besar = jalankan_ocr(citra_hasil)

        jumlah_benar, persentase = evaluasi_ocr(nomor_ocr)

        simpan_citra(
            citra_besar,
            folder_gambar,
            metode
        )

        data_gambar.append({
            "Gambar": file_gambar.name,
            "Metode": metode,
            "Hasil OCR": nomor_ocr,
            "Karakter Benar": jumlah_benar,
            "Akurasi (%)": round(persentase, 2)
        })

        print(
            f"{metode:<22} | "
            f"OCR: {nomor_ocr:<18} | "
            f"Benar: {jumlah_benar:>2} | "
            f"Akurasi: {persentase:6.2f}%"
        )

    return data_gambar


# ============================================================
# MEMBUAT LAPORAN DAN RINGKASAN
# ============================================================

def simpan_laporan(data_hasil):
    """Menyimpan detail hasil OCR dan ringkasan metode."""

    if not data_hasil:
        print("\nTidak ada hasil OCR yang bisa disimpan.")
        return

    tabel = pd.DataFrame(data_hasil)

    file_detail = FOLDER_OUTPUT / "hasil_ocr_detail.csv"

    tabel.to_csv(
        file_detail,
        index=False,
        encoding="utf-8-sig"
    )

    # Hitung jumlah gambar yang berhasil diproses
    jumlah_gambar = tabel["Gambar"].nunique()

    ringkasan = (
        tabel.groupby("Metode")
        .agg(
            **{
                "Karakter Benar": ("Karakter Benar", "sum"),
                "Akurasi Rata-rata (%)": ("Akurasi (%)", "mean")
            }
        )
        .reset_index()
    )

    ringkasan["Total Karakter"] = (
        len(NOMOR_TARGET) * jumlah_gambar
    )

    ringkasan["Akurasi Total (%)"] = (
        ringkasan["Karakter Benar"]
        / ringkasan["Total Karakter"]
        * 100
    ).round(2)

    ringkasan["Akurasi Rata-rata (%)"] = (
        ringkasan["Akurasi Rata-rata (%)"].round(2)
    )

    ringkasan = ringkasan[
        [
            "Metode",
            "Karakter Benar",
            "Total Karakter",
            "Akurasi Rata-rata (%)",
            "Akurasi Total (%)"
        ]
    ]

    file_ringkasan = (
        FOLDER_OUTPUT / "hasil_ocr_ringkasan.csv"
    )

    ringkasan.to_csv(
        file_ringkasan,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("RINGKASAN HASIL OCR")
    print("=" * 70)
    print(ringkasan.to_string(index=False))

    print("\nFile laporan berhasil disimpan:")
    print(f"- {file_detail}")
    print(f"- {file_ringkasan}")


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    print("=" * 70)
    print("PROGRAM FILTERING, NOISE REDUCTION, SHARPENING DAN OCR")
    print("=" * 70)

    print(f"Nomor target : {NOMOR_TARGET}")
    print(f"Folder input : {FOLDER_INPUT}")
    print(f"Folder output: {FOLDER_OUTPUT}")

    daftar_file = sorted(
        [
            *FOLDER_INPUT.glob("*.jpg"),
            *FOLDER_INPUT.glob("*.jpeg"),
            *FOLDER_INPUT.glob("*.png")
        ]
    )

    if not daftar_file:
        print("\nTidak ada gambar ditemukan di folder gambar.")
        print("Pastikan file gambar sudah berada di folder tersebut.")
        return

    print(f"Jumlah file ditemukan: {len(daftar_file)}")

    seluruh_hasil = []

    for file_gambar in daftar_file:
        hasil = proses_satu_gambar(file_gambar)
        seluruh_hasil.extend(hasil)

    simpan_laporan(seluruh_hasil)

    print("\n" + "=" * 70)
    print("PEMROSESAN SELESAI")
    print(f"Gambar ditemukan       : {len(daftar_file)}")
    print(f"Data OCR yang dihasilkan: {len(seluruh_hasil)}")
    print(f"Semua hasil tersimpan di: {FOLDER_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
