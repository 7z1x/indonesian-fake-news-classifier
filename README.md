# 📰 Klasifikasi Berita Hoax Bahasa Indonesia

Membangun model untuk memisahkan berita hoax dari berita benar berbahasa Indonesia — dan mengukur seberapa jauh model ini benar-benar lebih baik daripada menebak "hoax" untuk semua berita.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Sastrawi](https://img.shields.io/badge/NLP-Sastrawi-green)](https://github.com/har07/PySastrawi)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Indonesia%20False%20News-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews)

**📓 Notebook:** [Google Colab](https://colab.research.google.com/drive/1GVIg4lioHsMq_R0P0AFrEkO2dzvJ3YXM) · **📦 Dataset:** [Kaggle — Indonesia False News](https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews)

---

## 📌 Ringkasan Eksekutif

| Pertanyaan | Jawaban |
|---|---|
| Apa masalahnya? | Memisahkan **berita hoax (81,9%)** dari **berita benar (18,1%)** berdasarkan teks judul + narasi |
| Model terbaik? | **SVM** (kernel ter-tuning) — akurasi **84,18%**, F1 kelas hoax 0,91 |
| Seberapa andal? | **Kurang.** Baseline "selalu tebak hoax" sudah mencapai **81,94%** |
| Jadi sebenarnya apa yang dipelajari model? | **Hampir hanya prior kelas.** SVM menandai **94,7%** berita uji sebagai hoax |
| Di mana kelemahan terbesarnya? | **Berita benar nyaris tidak terdeteksi** — 121 dari 153 artikel asli salah dilabeli hoax |

> **Insight utama:** angka akurasi 84% terlihat meyakinkan sampai dibandingkan dengan baseline yang tidak melakukan apa pun. Dataset ini 81,9% hoax — artinya model yang **selalu** menjawab "hoax" sudah mendapat 81,94% akurasi tanpa membaca satu kata pun. SVM hanya unggul **2,2 poin persentase** di atas garis itu, sambil menuduh 94,7% artikel sebagai hoax. Untuk keperluan moderasi konten, model ini **belum layak dipakai**: ia akan menandai hampir semua berita asli sebagai palsu.

---

## 🎯 Business Understanding

### Latar Belakang

Berita palsu telah menjadi isu global yang signifikan, terutama di Indonesia sebagai salah satu negara dengan pengguna media sosial terbesar di dunia. Penyebaran informasi yang salah dapat memicu kepanikan sosial, memanipulasi opini publik, dan menimbulkan kerugian ekonomi [1]. Deteksi berita palsu dalam bahasa Indonesia memerlukan pendekatan khusus karena karakteristik morfologi dan konteks budayanya berbeda dari bahasa Inggris.

Secara operasional, moderasi konten adalah masalah **antrean terbatas**. Tim moderasi hanya mampu meninjau sebagian kecil konten setiap hari, sehingga kesalahan punya dua harga yang berbeda:

- **False negative** (hoax lolos sebagai berita benar) → informasi palsu menyebar ke publik.
- **False positive** (berita benar dituduh hoax) → pembatasan konten yang sah, hilangnya kepercayaan pengguna, dan risiko reputasi.

Pada dataset ini, kesalahan kedua justru **jauh lebih sering terjadi** daripada yang pertama — lihat [Hasil Model](#-hasil-model).

### Problem Statement

1. Bagaimana mengidentifikasi apakah suatu berita berbahasa Indonesia tergolong hoax atau benar secara otomatis?
2. Algoritma mana yang lebih baik untuk tugas ini: SVM atau Random Forest?
3. Apakah model yang dibangun benar-benar menambah nilai di atas tebakan naif?

### Goals

- Membangun model klasifikasi teks berbahasa Indonesia dengan akurasi minimal di atas baseline naif.
- Menangani ketidakseimbangan kelas (81,9% : 18,1%) agar kelas minoritas tetap terdeteksi.
- Membandingkan **SVM** dan **Random Forest** dengan hyperparameter tuning yang setara.

### Metrik Keberhasilan

- **Metrik bisnis:** proporsi berita benar yang salah ditandai (**false positive rate**) — karena inilah biaya yang paling terasa pada pengguna.
- **Metrik model:** **Recall kelas minoritas (berita benar)** dan **F1 per kelas**, bukan akurasi. Dengan kelas mayoritas 81,9%, akurasi tinggi bisa dicapai tanpa model belajar apa pun. Baseline naif dicantumkan secara eksplisit sebagai pembanding.

### Cakupan

**Termasuk:** EDA, preprocessing teks bahasa Indonesia, ekstraksi fitur TF-IDF, penanganan imbalance, tuning 2 algoritma, evaluasi per kelas, analisis confusion matrix.
**Tidak termasuk:** analisis sitasi/sumber berita, verifikasi fakta berbasis basis data eksternal, analisis gambar pendukung, model deep learning (LSTM/GRU/IndoBERT), deployment API.

---

## 🗂️ Dataset

| Atribut | Keterangan |
|---|---|
| **Sumber** | [Kaggle — Indonesia False News](https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews) (Muhammad Ghazi Muharam) |
| **Ukuran awal** | **4.701** baris × 6 kolom |
| **Kolom** | `ID`, `label`, `tanggal`, `judul`, `narasi`, `nama file gambar` |
| **Label kosong** | **470** baris (**10,0%**) |
| **Data terpakai** | **4.231** baris |
| **Target** | `label` — `1` = hoax, `0` = berita benar |

**Distribusi target:**

| Label | Keterangan | Jumlah | Proporsi |
|---|---|---:|---:|
| 1 | Berita Hoax | **3.465** | **81,9%** |
| 0 | Berita Benar | **766** | **18,1%** |

Rasio 4,5 : 1 ini adalah fakta paling menentukan dalam seluruh proyek. Setiap metrik harus dibaca dengan latar angka ini.

**Catatan kualitas data — label kosong bukan hilang acak.** Ke-470 baris berlabel `NaN` menempati **blok terakhir** dataset (indeks 4.231–4.700), bukan tersebar. Artinya ini bukan kerusakan data acak, melainkan data yang memang tidak pernah dilabeli di bagian akhir berkas. Penghapusan baris tersebut aman untuk distribusi, tetapi menunjukkan bahwa dataset sumbernya kemungkinan hasil penggabungan dua berkas dengan kelengkapan berbeda.

| Struktur Awal | Blok Label Kosong |
|---|---|
| ![Data awal](image/struktur%20data%20awal.jpg) | ![Label kosong](image/null%20data.jpg) |

| Setelah Dropna | Kolom Teks Gabungan |
|---|---|
| ![Bersih](image/data%20setelah%20dropna%20label%20null.jpg) | ![Teks gabungan](image/data%20teks%20gabungan%20judul%20dan%20narasi.jpg) |

---

## 🔬 Metodologi

```
Data Loading → Gabung Train+Test → EDA → Drop Label Kosong → Gabung Judul+Narasi
→ Preprocessing Teks → Split 80:20 → TF-IDF → SMOTE → Modeling → Tuning → Evaluasi
```

1. **Data Loading** — `Data_latih.csv` + `Data_uji.csv` digabung menjadi satu, karena pembagian ulang dilakukan sendiri dengan stratifikasi.
2. **EDA** — cek struktur, missing value, distribusi label, dan distribusi panjang teks per kelas.
3. **Cleaning** — hapus 470 baris dengan `label` kosong.
4. **Penggabungan kolom** — `judul` + `narasi` → satu kolom `teks`, agar model melihat konteks penuh.
5. **Preprocessing teks** — lowercase, hapus angka (regex `\d+`), hapus tanda baca (`string.punctuation`), hapus stopword bahasa Indonesia, lalu **stemming** dengan Sastrawi. Dioptimasi dengan `@lru_cache` dan `set` untuk stopword, karena stemming 4.231 dokumen panjang secara naif sangat lambat.
6. **Split** — 80:20 **stratified** (`random_state=42`) → **3.384 train / 847 test**.
7. **Ekstraksi fitur** — **TF-IDF** `max_features=1000`, `ngram_range=(1,2)` (unigram + bigram). Di-*fit* pada data latih saja, lalu di-*transform* ke data uji.
8. **SMOTE** — data latih diseimbangkan menjadi **2.771 : 2.771**. Data uji **tidak** disentuh.
9. **Modeling** — SVM dan Random Forest, masing-masing dengan `RandomizedSearchCV` (`cv=5`, `n_iter=10`, `scoring='accuracy'`).
10. **Evaluasi** — classification report, confusion matrix, cross-validation score, dan pembandingan terhadap baseline naif.

**Kenapa TF-IDF dengan bigram?** Unigram menangkap kata kunci, bigram menangkap frasa yang bermakna utuh (misalnya "kabar bohong", "tidak benar"). `max_features=1000` membatasi dimensi agar model tetap ringan dan mengurangi noise dari kata yang sangat jarang.

**Kenapa dua algoritma?** SVM linear secara klasik kuat untuk data teks berdimensi tinggi dan sparse. Random Forest diuji sebagai pembanding non-linear yang lebih tahan noise. Perbandingannya menunjukkan keduanya menghadapi masalah yang sama — lihat [Hasil Model](#-hasil-model).

---

## 📊 Temuan Utama (EDA)

### 1. Dataset sangat tidak seimbang — ini menentukan segalanya

| Kelas | Jumlah | Proporsi |
|---|---:|---:|
| Hoax | 3.465 | **81,9%** |
| Benar | 766 | **18,1%** |

Dengan komposisi ini, **akurasi bukan metrik yang berguna**. Model yang mengabaikan teks sama sekali dan selalu menjawab "hoax" sudah mendapat **81,94%** pada data uji. Angka ini dipakai sebagai garis pembanding di seluruh laporan ini.

![Distribusi Label](image/distribusi%20data%20label.jpg)

### 2. Panjang teks hampir tidak membedakan kelas

Distribusi panjang teks untuk hoax dan berita benar **sangat mirip**: keduanya memuncak di kisaran 100–200 karakter dan berekor panjang hingga ~1.400 karakter. Tidak ada pemisahan visual yang jelas.

Ini informasi negatif yang berguna: **panjang teks bukan sinyal**, sehingga model harus bergantung pada pilihan kata, bukan pada seberapa panjang artikelnya. Kalau sebaliknya (misalnya hoax selalu jauh lebih pendek), model berisiko hanya mempelajari artefak panjang dokumen.

![Panjang Teks](image/panjang%20teks.jpg)

---

## 🤖 Hasil Model

### Perbandingan performa

| Model | Accuracy | Precision (benar) | Recall (benar) | F1 (hoax) | CV Score |
|---|---:|---:|---:|---:|---:|
| **SVM** | **0,8418** | 0,71 | **0,21** | 0,91 | 0,9264 |
| **Random Forest** | 0,8394 | 0,58 | **0,42** | 0,90 | 0,9142 |
| *Baseline "selalu hoax"* | *0,8194* | — | *0,00* | *0,90* | — |

**Inilah inti persoalannya.** SVM "menang" akurasi, tetapi:

- **Recall kelas benar hanya 0,21** — model hanya menemukan 32 dari 153 artikel asli.
- **Selisih terhadap baseline naif hanya 2,2 poin persentase** (84,18% vs 81,94%).
- Random Forest lebih baik dalam mengenali berita benar (recall 0,42) meski akurasi totalnya sedikit lebih rendah.

### Confusion matrix — angka yang sebenarnya berarti

Test set: **847 artikel** (153 benar, 694 hoax).

| Model | TN | FP | FN | TP | Artikel ditandai hoax |
|---|---:|---:|---:|---:|---:|
| **SVM** | 32 | **121** | 13 | 681 | **802 / 847 (94,7%)** |
| **Random Forest** | 65 | **88** | 50 | 644 | 733 / 847 (86,6%) |

| Confusion Matrix SVM | Confusion Matrix Random Forest |
|---|---|
| ![SVM](image/confusion%20matrix%20svm.jpg) | ![RF](image/confusion%20matrix%20random%20forest.jpg) |

**Cara membaca untuk konteks moderasi konten:**

- SVM menandai **94,7%** dari seluruh artikel sebagai hoax. Model ini praktis berperilaku seperti filter yang hampir selalu menyala.
- Dari 153 berita **benar**, SVM hanya meloloskan **32** dan menuduh **121** sebagai hoax. Bagi pengguna, ini berarti artikel yang sah pun ikut diturunkan jangkauannya.
- Random Forest menuduh 88 berita benar sebagai hoax — masih buruk, tapi **tidak separah SVM**. Ia juga lebih seimbang: melewatkan 50 hoax, bukan 13.
- SVM lebih unggul dalam **menangkap hoax** (681 vs 644), tapi keunggulan itu dibeli dengan mengorbankan berita asli secara masif.

### Cross-validation vs test — jarak 8,5 poin yang perlu dijelaskan

| Model | CV Score (5-fold) | Test Accuracy | Selisih |
|---|---:|---:|---:|
| SVM | 0,9264 | 0,8418 | **−8,5 poin** |
| Random Forest | 0,9142 | 0,8394 | **−7,5 poin** |

Jarak sebesar ini bukan kebetulan. Penyebabnya struktural: `RandomizedSearchCV` dijalankan **di atas data yang sudah di-SMOTE**, sehingga setiap fold validasi memuat sampel sintetis yang dibangkitkan dari tetangga terdekat sampel di fold latih. Sampel sintetis itu **bocor** ke validasi, dan skor CV menjadi terlalu optimistis. Ini dibahas di [Keterbatasan](#-keterbatasan--asumsi) poin 2.

**Catatan transparansi:** seluruh angka di atas dihitung ulang dari confusion matrix yang tersimpan sebagai gambar di repo ini, dan konsisten dengan classification report pada notebook (selisih pembulatan < 0,3 poin). Nilai persis untuk *setiap* kombinasi hyperparameter tidak tersimpan, karena output sel notebook tidak di-*commit*.

---

## ⚠️ Keterbatasan & Asumsi

Tiga poin pertama adalah **masalah metodologis yang memengaruhi kesimpulan**, bukan catatan kecil.

### 1. Model hampir tidak melampaui baseline naif

Ini temuan terpenting dari seluruh proyek, dan seharusnya menjadi kalimat pertama di laporan mana pun yang memakai hasil ini. Akurasi 84,18% terdengar baik sampai dibandingkan dengan 81,94% yang bisa dicapai tanpa model sama sekali. Kenaikan 2,2 poin itu sendiri **belum tentu signifikan** secara statistik pada 847 sampel.

### 2. SMOTE pada TF-IDF, lalu CV di atas hasil SMOTE

Dua masalah bertumpuk:

- **SMOTE tidak cocok untuk data teks sparse.** SMOTE membangkitkan sampel sintetis dengan interpolasi linear antar tetangga. Pada TF-IDF 1.000 dimensi, hasil interpolasi bukan dokumen yang masuk akal — ia menghasilkan vektor dengan kombinasi kata yang tidak pernah muncul bersamaan. Untuk teks, penanganan imbalance yang lebih tepat adalah `class_weight='balanced'` atau penyesuaian threshold, bukan oversampling sintetis.
- **CV dijalankan setelah SMOTE.** Idealnya SMOTE berada **di dalam** pipeline, sehingga hanya diterapkan pada fold latih setiap iterasi. Seperti diimplementasikan sekarang, skor CV (0,9264) tidak bisa dipakai sebagai estimasi performa sebenarnya.

### 3. Model dipilih berdasarkan akurasi data uji

Notebook memilih model terbaik dengan membandingkan akurasi pada **test set**. Ini membuat test set tidak lagi netral: model yang dilaporkan sebagai "terbaik" sudah terpilih karena cocok dengan data uji tersebut. Pemilihan model seharusnya dilakukan pada data validasi terpisah atau lewat CV.

### 4. `scoring='accuracy'` pada dataset 82:18

`RandomizedSearchCV` dioptimasi untuk akurasi. Pada data timpang, ini secara matematis mendorong model ke arah memprediksi kelas mayoritas — persis perilaku yang terlihat di confusion matrix. Metrik yang seharusnya dipakai: **F1-macro**, **balanced accuracy**, atau **recall kelas minoritas**.

### 5. Label "benar" belum tentu benar

Dataset ini memberi label biner, tetapi tidak menyertakan **sumber verifikasi** (apakah dicek ke situs fact-check, apakah ada tanggal verifikasi). Berita yang dilabeli "benar" bisa saja merupakan klaim yang belum terbukti — hanya belum terbukti salah. Model mempelajari label apa adanya, termasuk potensi kesalahan pelabelan di dalamnya.

### 6. Tidak ada pengujian pada berita di luar periode data

Data mencakup rentang tanggal 2020 dan model tidak diuji pada berita setelah periode itu. Gaya bahasa hoax berubah mengikuti isu yang beredar, sehingga performa pada berita baru **belum diketahui** dan bisa turun.

### 7. Fitur dibatasi 1.000 kata

`max_features=1000` membuang sebagian besar kosakata. Untuk domain berita, kata yang jarang tapi sangat menandakan hoax (nama tokoh, istilah politik tertentu) bisa ikut terbuang. Batas ini dipilih demi efisiensi, bukan karena optimal.

### 8. Teks hanya judul + narasi

Nama file gambar tersedia di dataset tetapi tidak dipakai. Pada praktiknya, banyak hoax dikenali dari gambar yang didaur ulang dari konteks lain — sinyal yang sepenuhnya hilang di model ini.

---

## 💡 Rekomendasi

### Untuk perbaikan model (prioritas utama)

1. **Ganti SMOTE dengan `class_weight='balanced'`** dan bandingkan hasilnya. Ini lebih tepat untuk data teks dan menghilangkan artefak sampel sintetis.
2. **Bungkus seluruh proses dalam `Pipeline`** (TF-IDF → SMOTE/class_weight → model) agar CV mengevaluasi alur yang benar dan tidak ada kebocoran antar fold.
3. **Ubah metrik tuning** dari `accuracy` menjadi `f1_macro` atau `recall` kelas minoritas.
4. **Optimalkan threshold keputusan.** Probabilitas SVM/RF hampir selalu condong ke hoax. Dengan threshold yang lebih tinggi, model bisa diarahkan untuk lebih selektif menandai — dan trade-off-nya bisa diukur langsung pada kurva PR.
5. **Laporkan baseline di setiap evaluasi.** Setiap tabel hasil sebaiknya menyertakan baris "selalu prediksi kelas mayoritas" sebagai pembanding. Tanpa itu, angka 84% menyesatkan.
6. **Uji model berbasis transformer bahasa Indonesia** (IndoBERT). Untuk teks, representasi kontekstual biasanya jauh melampaui TF-IDF — dan perbandingannya akan menunjukkan apakah batas performa di sini berasal dari metode atau dari data.

### Untuk implementasi (dengan syarat)

1. **Jangan pakai model ini untuk keputusan otomatis.** Gunakan sebagai **alat bantu prioritaskan antrean** — model menyortir artikel yang perlu ditinjau manusia lebih dulu, keputusan tetap di tangan moderator.
2. **Sesuaikan threshold dengan kapasitas tim.** Jika tim hanya sanggup meninjau 200 artikel per hari, threshold sebaiknya menghasilkan sekitar 200 kandidat dengan presisi tertinggi — bukan memaksimalkan recall sampai 95%.
3. **Ukur kedua jenis kesalahan secara terpisah.** Laporkan berapa berita benar yang salah ditandai per minggu, bukan hanya akurasi agregat.
4. **Bangun mekanisme umpan balik.** Catat keputusan moderator atas setiap artikel yang ditandai, lalu pakai sebagai data berlabel baru untuk iterasi berikutnya.

### KPI yang Disarankan

| KPI | Baseline (test set) | Target |
|---|---|---|
| Akurasi di atas baseline naif | +2,2 poin | **+10 poin** |
| Recall kelas berita benar | 0,21 (SVM) / 0,42 (RF) | **≥ 0,75** |
| False positive per 100 berita benar | 79 (SVM) | **< 15** |
| Selisih CV vs test | 8,5 poin | **< 2 poin** |
| Presisi pada 200 kandidat teratas | Belum diukur | **≥ 0,90** |

---

## 🚀 Cara Menjalankan

### 1. Prasyarat

- Python **3.10+**
- Akun Kaggle + `kaggle.json` (notebook mengunduh dataset via Kaggle API)

### 2. Install dependensi

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn Sastrawi tqdm
```

### 3. Siapkan dataset

```bash
kaggle datasets download -d muhammadghazimuharam/indonesiafalsenews
unzip indonesiafalsenews.zip
```

### 4. Jalankan

**Opsi A — Notebook (disarankan):**

```bash
jupyter notebook "ProjectPertama_predictiveAnalytic(A548YBM523)_FInal.ipynb"
```

**Opsi B — Script Python:**

```bash
python "Projectpertama_predictiveanalytic(a548ybm523)_Final.py"
```

> **Catatan:** script `.py` hasil ekspor Colab masih memuat `files.upload()`, `!kaggle`, dan `!unzip` yang hanya jalan di Colab, serta path absolut `/content/Data_latih.csv`. Untuk eksekusi lokal, unduh dataset manual lalu sesuaikan path `read_csv`.

---

## 🗂️ Struktur Proyek

```
indonesian-fake-news-classifier/
│
├── ProjectPertama_predictiveAnalytic(A548YBM523)_FInal.ipynb   # Notebook utama
├── Projectpertama_predictiveanalytic(a548ybm523)_Final.py      # Ekspor Colab
├── README.md
│
└── image/
    ├── struktur data awal.jpg                        # Struktur awal (4.701 × 6)
    ├── null data.jpg                                 # Blok 470 label kosong
    ├── data setelah dropna label null.jpg            # Setelah dropna (4.231 × 6)
    ├── data teks gabungan judul dan narasi.jpg       # Kolom teks + label
    ├── distribusi data label.jpg                     # Distribusi label
    ├── panjang teks.jpg                              # Distribusi panjang teks
    ├── confusion matrix svm.jpg                      # Confusion matrix SVM
    └── confusion matrix random forest.jpg            # Confusion matrix RF
```

---

## 🛠️ Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3.10+ |
| Analisis data | pandas, NumPy |
| NLP | Sastrawi (stopword + stemming bahasa Indonesia), regex |
| Ekstraksi fitur | TF-IDF (scikit-learn) |
| Machine learning | scikit-learn (SVM, Random Forest, RandomizedSearchCV) |
| Penanganan imbalance | imbalanced-learn (SMOTE) |
| Visualisasi | Matplotlib, Seaborn |
| Environment | Jupyter / Google Colab |

---

## 📚 Referensi

[1] Kurniawan, A. A., & Mustikasari, M. (2021). Implementasi deep learning menggunakan metode CNN dan LSTM untuk menentukan berita palsu dalam bahasa Indonesia. *Jurnal Informatika Universitas Pamulang*, 5(4), 544–552. https://doi.org/10.32493/informatika.v5i4.6760

---

## 👤 Author

**Zulfahmi M. Ardianto**
Laskar ID: A548YBM523

*Project ini merupakan bagian dari portfolio Data Analyst.*
