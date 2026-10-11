print("=== PROGRAM PERKALIAN MATRIKS ARRAY 3 DIMENSI ===")

jumlah_matriks = int(input("Masukkan jumlah matriks: "))

# INPUT UKURAN MATRIKS A

baris_A = int(input("\nMasukkan jumlah baris Matriks A: "))
kolom_A = int(input("Masukkan jumlah kolom Matriks A: "))

# INPUT UKURAN MATRIKS B

while True:

    baris_B = int(input("\nMasukkan jumlah baris Matriks B: "))
    kolom_B = int(input("Masukkan jumlah kolom Matriks B: "))

    if kolom_A == baris_B:
        print("\nMatriks bisa dikalikan!")
        break

    else:
        print("\nMatriks tidak bisa dikalikan!")
        print("Jumlah kolom Matriks A harus sama dengan jumlah baris Matriks B.")
        print("Silakan masukkan ukuran Matriks B lagi.")


# INPUT ARRAY 3D A

A = []

print("\n=== INPUT MATRIKS A ===")

for m in range(jumlah_matriks):

    print(f"\nMatriks A ke-{m + 1}")

    matriks = []

    for i in range(baris_A):

        while True:

            baris = list(
                map(
                    int,
                    input(f"Masukkan baris ke-{i + 1}: ").split()
                )
            )

            if len(baris) == kolom_A:
                break
            else:
                print(f"Harus memasukkan {kolom_A} angka!")

        matriks.append(baris)

    A.append(matriks)

# INPUT ARRAY 3D B

B = []

print("\n=== INPUT MATRIKS B ===")

for m in range(jumlah_matriks):

    print(f"\nMatriks B ke-{m + 1}")

    matriks = []

    for i in range(baris_B):

        while True:

            baris = list(
                map(
                    int,
                    input(f"Masukkan baris ke-{i + 1}: ").split()
                )
            )

            if len(baris) == kolom_B:
                break
            else:
                print(f"Harus memasukkan {kolom_B} angka!")

        matriks.append(baris)

    B.append(matriks)

# PROSES PERKALIAN

hasil_3D = []

for m in range(jumlah_matriks):

    hasil_matriks = []

    for i in range(baris_A):

        baris_hasil = []

        for j in range(kolom_B):

            hasil = 0

            for k in range(kolom_A):

                hasil += A[m][i][k] * B[m][k][j]

            baris_hasil.append(hasil)

        hasil_matriks.append(baris_hasil)

    hasil_3D.append(hasil_matriks)

# HASIL

print("\n=== HASIL PERKALIAN ===")

for m in range(jumlah_matriks):

    print(f"\nMatriks hasil ke-{m + 1}")

    for baris in hasil_3D[m]:
        print(baris)


print("\nHasil Array 3 Dimensi:")
print(hasil_3D)