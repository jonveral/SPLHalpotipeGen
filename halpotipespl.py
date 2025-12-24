import numpy as np

# DATASET GENETIK HALPOTIPE
# Urutan sifat: [Ganteng, Cantik, Tinggi, Pintar]
sifat = ["Ganteng", "Cantik", "Tinggi", "Pintar"]
bapak = [2, 0, 1, 2]
mamak = [0, 2, 2, 1]
kakek = [1, 0, 2, 1]
nenek = [0, 1, 1, 2]
anak  = [1, 0, 1, 1]

# TAMPILKAN DATASET
print("DATASET HAPLOTIPE GENETIK")
print(f"{'Individu':<8} | {' '.join([f'{s:<8}' for s in sifat])}")
data = {
    "Pak": bapak,
    "Mak": mamak,
    "Kakek": kakek,
    "Nenek": nenek,
    "Anak": anak
}
for k, v in data.items():
    print(f"{k:<8} | {' '.join([f'{x:<8}' for x in v])}")

# PEMBENTUKAN MATRIKS SPL
A = np.array([bapak, mamak, kakek, nenek], dtype=float)
b = np.array(anak, dtype=float).reshape(-1, 1)
aug = np.hstack((A, b))
n = len(b)

print("\n")
print("HAPLOTIPE GENETIK (GAUSS–JORDAN)")
print("Matriks A (Orang Tua):")
print(A)
print("\nVektor b (Anak):")
print(b.flatten())

# GAUSS–JORDAN
for i in range(n):
    if aug[i][i] == 0:
        for k in range(i + 1, n):
            if aug[k][i] != 0:
                aug[[i, k]] = aug[[k, i]]
                break
    aug[i] = aug[i] / aug[i][i]
    for j in range(n):
        if j != i:
            aug[j] = aug[j] - aug[j][i] * aug[i]
print("\n")
print("HASIL GAUSS–JORDAN (RREF)")
print(np.round(aug, 3))

# KOEFISIEN SPL (PERSEN MENTAH)
x = aug[:, -1]
label = ["Pak", "Mak", "Kakek", "Nenek"]
print("\n")
print("KOEFISIEN HASIL SPL (PERSEN MENTAH)")
for l, v in zip(label, x):
    print(f"{l:<7}: {v*100:6.2f}%")

# NORMALISASI
total = np.sum(np.abs(x))
x_norm = np.abs(x) / total
print("\n")
print("KONTRIBUSI SETELAH NORMALISASI (%)")
for l, v in zip(label, x_norm):
    print(f"{l:<7}: {v*100:6.2f}%")