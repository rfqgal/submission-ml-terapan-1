# Laporan Proyek Machine Learning - Rifky Galuh Yuliawan

## Domain Proyek

Pendapatan seseorang dapat dipengaruhi oleh berbagai faktor sosial-ekonomi seperti usia, tingkat pendidikan, jenis kelamin, status perkawinan, pekerjaan, dan ukuran pemukiman. Dengan memahami hubungan antara faktor-faktor ini dan pendapatan dapat membantu dalam perencanaan keuangan, kebijakan sosial, serta pengembangan program peningkatan kesejahteraan masyarakat. Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi pendapatan individu berdasarkan faktor sosial-ekonomi, sehingga memberikan wawasan yang berguna bagi pengambil keputusan.

### Mengapa Masalah Ini Penting dan Bagaimana Harus Diselesaikan?

Prediksi pendapatan berbasis data dapat membantu pemerintah dalam merancang kebijakan yang lebih tepat sasaran, misalnya program bantuan sosial atau pelatihan kerja. Dengan menggunakan pendekatan machine learning, kita dapat mengidentifikasi pola yang tidak terlihat secara manual dan memberikan prediksi yang lebih akurat dibandingkan metode tradisional.

### Referensi

Studi serupa telah dilakukan oleh peneliti terkait untuk menganalisis hubungan antara faktor sosial-ekonomi dan kondisi ekonomi sebagai berikut:

- Avduevskaya E., Nadezhina O., Zaborovskaia O., 2023. The Impact of Socio-Economic Factors on the Regional Economic Security Indicator. International Journal of Technology. Volume 14(8), pp. 1706-1716

## Business Understanding

### Problem Statements

- Bagaimana cara memprediksi pendapatan individu berdasarkan faktor sosial-ekonomi seperti usia, pendidikan, pekerjaan, dan lainnya dengan akurasi yang tinggi?
- Model machine learning mana yang paling efektif untuk memprediksi pendapatan berdasarkan dataset sosial-ekonomi yang tersedia?

### Goals

- Membangun model machine learning yang dapat memprediksi pendapatan individu dengan nilai Mean Squared Error (MSE) yang rendah.
- Membandingkan performa beberapa algoritma machine learning (K-Nearest Neighbors, Random Forest, dan AdaBoost) untuk menentukan model terbaik dalam memprediksi pendapatan.

### Solusion Statement

- Menggunakan tiga algoritma machine learning, yaitu K-Neighbors Regressor, Random Forest Regressor, dan AdaBoost Regressor, untuk memprediksi pendapatan, lalu membandingkan performa mereka berdasarkan metrik evaluasi MSE.
- Melakukan hyperparameter tuning pada ketiga algoritma untuk meningkatkan performa model, diukur dengan penurunan nilai MSE.

## Data Understanding

Proyek ini menggunakan dataset "Socioeconomic Factors and Income Dataset" yang tersedia di Kaggle [2]. Dataset ini berisi 2000 entri dengan informasi faktor sosial-ekonomi individu beserta pendapatan tahunan mereka. Dataset dapat diunduh dari tautan berikut: https://www.kaggle.com/datasets/aldol07/socioeconomic-factors-and-income-dataset/data.

### Variabel-variabel pada Dataset:

- ID: Identifikasi unik untuk setiap individu (numerik).
- Sex: Jenis kelamin (0 untuk laki-laki, 1 untuk perempuan).
- Marital status: Status perkawinan (single atau non-single: divorced/separated/married/widowed).
- Age: Usia individu (numerik, rentang 18–76 tahun).
- Education: Tingkat pendidikan (other, high school, university, graduate school).
- Income: Pendapatan tahunan individu (numerik, target variable).
- Occupation: Jenis pekerjaan (unemployed/unskilled, skilled employee/official, management/self-employed/highly qualified).
- Settlement size: Ukuran pemukiman (0: kecil, 1: sedang, 2: besar).

### Exploratory Data Analysis (EDA):

#### Missing Values

![Missing Values](https://raw.githubusercontent.com/rfqgal/submission-ml-terapan-1/refs/heads/master/images/missing-values.png)

Pada tahap analisis nilai kosong atau hilang, dapat diketahui bahwa **tidak ditemukan** adanya nilai yang hilang pada dataset.

#### Duplicate Data

![Duplicates](https://raw.githubusercontent.com/rfqgal/submission-ml-terapan-1/refs/heads/master/images/duplicates.png)

Pada tahap pengecekan data duplikat, juga **tidak ditemukan** adanya data duplikat pada dataset.

#### Outliers

![Outliers](https://raw.githubusercontent.com/rfqgal/submission-ml-terapan-1/refs/heads/master/images/outliers.png)

Pada tahap analisis outliers, diketahui terdapat outliers pada Income dan Age yang jumlahnya mencapai 130. Jika dibandingkan dengan jumlah keseluruhan data, yaitu 2000, maka tidak masalah jika seluruh **outliers dibuang** dari dataset.

#### Analisis Tambahan

Untuk memahami data lebih lanjut, dilakukan visualisasi relativitas variabel kategorikal dengan variabel target (Income) menggunakan `catplot`. Berikut visualisasi yang dihasilkan:

![Income-Occupation](https://raw.githubusercontent.com/rfqgal/submission-ml-terapan-1/refs/heads/master/images/income-occupation.png)

Distribusi pendapatan menunjukkan bahwa sebagian besar individu memiliki pendapatan di bawah 150.000, yaitu masyarakat dengan Occupation "skilled employee / official" dan yang paling rendah "unemployed / unskilled".

Selain itu, korelasi antar variabel numerik dianalisis menggunakan heatmap:

![Heatmap](https://raw.githubusercontent.com/rfqgal/submission-ml-terapan-1/refs/heads/master/images/heatmap.png)

Heatmap menunjukkan bahwa usia dan pendapatan memiliki korelasi positif yang moderat, sementara variabel lain seperti Settlement size juga memiliki pengaruh tertentu.

## Data Preparation

Tahap data preparation dilakukan untuk memastikan data siap digunakan dalam pemodelan. Berikut adalah langkah-langkah yang dilakukan beserta alasannya:

### 1. Menangani Outliers

Proses ini langsung dilakukan di awal ketika ditemukannya outliers.

```
Q1 = numerical_features.quantile(0.25)
Q3 = numerical_features.quantile(0.75)
IQR = Q3 - Q1

filter_outliers = ~((numerical_features < (Q1 - 1.5 * IQR)) |
                    (numerical_features > (Q3 + 1.5 * IQR))).any(axis=1)

df_filtered = df[filter_outliers]
df_filtered.shape
```

#### Penjelasan Proses

Tahap ini dilakukan dengan metode Interquartile Range (IQR) untuk memastikan kualitas data sebelum pemodelan. Proses dimulai dengan menghitung kuartil pertama (Q1) dan kuartil ketiga (Q3) dari fitur numerik, lalu menghitung IQR sebagai selisih Q3 dan Q1. Batas bawah dan atas ditentukan dengan rumus `Q1 - 1.5 * IQR` dan `Q3 + 1.5 * IQR`, kemudian data yang berada di luar rentang ini diidentifikasi sebagai outliers. Filter dibuat untuk mengecualikan baris dengan outliers, dan DataFrame difilter sesuai filter tersebut, menghasilkan dataset bersih yang digunakan untuk analisis lebih lanjut.

#### Alasan Dilakukannya Proses

Penanganan outliers dengan cara menghilangkannya dipilih karena perbandingan jumlah outliers dan keseluruhan data cukup kecil, yaitu 130:2000. Dataset akhir yang berjumlah 1870 masih cukup banyak untuk implementasi proyek ini.

### 2. Menghapus Kolom ID

```
df_filtered = df_filtered.drop(['ID'], axis=1)
df_filtered.head()
```

#### Penjelasan Proses

DataFrame didefinisi ulang dengan menerapkan fungsi `drop()` yang berisi ID, yaitu nama kolom yang akan dihilangkan. Kemudian ditulis juga `axis=1` yang berarti perintah akan dijalankan pada keseluruhan deret kolom yang dimaksud.

#### Alasan Dilakukannya Proses

Fitur ID biasanya hanya merupakan identifikasi unik dari setiap baris data. Fitur ID ini memiliki korelasi yang sangat rendah, yaitu -0.38 terhadap variabel Income, sehingga tidak memberikan kontribusi signifikan dalam proyek prediksi pendapatan ini. Dengan menghapus fitur ID akan membantu mengurangi noise dan dimensi data, serta memastikan model fokus pada fitur yang relevan.

### 3. Data Splitting

```
X = df_filtered.drop(['Income'], axis=1)
y = df_filtered['Income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
```

#### Penjelasan Proses

Proses ini memisahkan dataset dalam perbandigan tertentu, biasanya 9:1 (seperti kode di atas) atau 8:2. Pembagian yang dilakukan menghasilkan 2 jenis variabel, yaitu train dan test, yang terdiri dari X (seluruh fitur selain Income) dan y (fitur Income). Sehingga proses ini akan menghasilkan 4 variabel dengan rincian sebagai berikut:

1. X_train: data X yang telah dipisah untuk proses pelatihan model.
2. X_test: data X untuk dilakukannya pengujian atau prediksi.
3. y_train: data y yang telah dipisah untuk proses pelatihan model.
4. y_test: data y untuk dilakukannya pengujian atau prediksi.

#### Alasan Dilakukannya Proses

Alasan dilakukannya proses ini untuk mempersiapkan masing-masing data yang dibutuhkan dan siap untuk digunakan dalam proses pelatihan dan pengujian. Data untuk pelatihan memerlukan perbandingan yang lebih banyak, karena proses pelatihan harus dilakukan dalam frekuensi yang lebih sering. Sedangkan data untuk pengujian hanya diperlukan dalam perbandingan yang kecil, karena proses pengujian tidak perlu dilakukan terlalu sering.

### 4. Feature Encoding & Scaling

```
preprocessor = ColumnTransformer([
    ('numerical', StandardScaler(), numerical_features.drop(['ID', 'Income'], axis=1).columns.to_list()),
    ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
])

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
```

#### Penjelasan Proses

Proses ini melakukan transformasi pada seluruh fitur yang digunakan, baik numerikal maupun kategorikal. Pada fitur numerikal dilakukan Scaling, sedangkan fitur kategorikal dilakukan Encoding.

#### Alasan Dilakukannya Proses

Alasan dilakukannya Scaling pada fitur numerikal adalah mengubah nilai pada setiap baris agar memiliki rentang nilai yang sama, sehingga fitur yang bernilai besar tidak mendominasi fitur yang bernilai lebih kecil. Sedangkan Encoding pada fitur kategorikal adalah mengubah nilainya menjadi format numerikal, sehingga bisa diproses oleh algoritma yang hanya menerima input berbasis angka.

## Modeling

Pemodelan machine learning untuk proyek ini menggunakan tiga algoritma Regressor: K-Neighbors, Random Forest, dan AdaBoost. Setiap model dioptimalkan menggunakan hyperparameter tuning dengan `GridSearchCV`.

### Tahap Penyusunan Model

Tahapan penyusunan model dimulai dengan persiapan data yang telah dilakukan sebelumnya, di mana fitur numerik diskalakan menggunakan StandardScaler dan fitur kategorikal diencode dengan OneHotEncoder melalui ColumnTransformer untuk memastikan kompatibilitas dengan algoritma. Selanjutnya, tiga algoritma regresi yakni K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost, didefinisi secara berurutan untuk melatih model dengan data train. Proses ini diikuti oleh evaluasi awal menggunakan Mean Squared Error (MSE) untuk membandingkan performa, kemudian dilakukan tuning hyperparameter menggunakan GridSearchCV untuk mengoptimalkan model. Akhirnya, model terbaik dipilih berdasarkan MSE terendah setelah tuning.

### Algoritma-algoritma yang Dipakai

#### **1. K-Neighbors Regressor**

- **Cara kerja**: Algoritma ini memprediksi dengan menghitung jarak antara data baru dan k tetangga terdekat dalam data latih, lalu mengambil rata-rata nilai target tetangga tersebut berdasarkan parameter k dan bobot yang ditentukan.

- **Parameter yang digunakan**

  ```
  (n_neighbors=5)
  ```
- **Kelebihan algoritma**: Mudah diimplementasikan dan tidak memerlukan asumsi distribusi data.
- **Kekurangan algoritma**: Sensitif terhadap skala data dan lambat pada dataset besar.

#### **2. Random Forest Regressor**

- **Cara kerja**: Algoritma ini membangun banyak pohon keputusan secara acak pada subset data dan fitur, kemudian menggabungkan prediksi semua pohon dengan rata-rata untuk meningkatkan akurasi dan mengurangi overfitting.

- **Parameter yang digunakan**

  ```
  (n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
  ```
- **Kelebihan algoritma**: Dapat menangani data dengan banyak fitur dan tidak mudah overfitting.
- **Kekurangan algoritma**: Membutuhkan lebih banyak sumber daya komputasi.

#### **3. AdaBoost Regressor**

- **Cara kerja**: Algoritma ini melatih model secara berurutan, dengan setiap model berikutnya fokus memperbaiki kesalahan sebelumnya dengan memberikan bobot lebih pada data yang salah diprediksi, yang diatur oleh learning rate dari parameternya.

- **Parameter yang digunakan**

  ```
  (learning_rate=0.05, random_state=55)
  ```
- **Kelebihan algoritma**: Baik untuk meningkatkan performa model yang lemah.
- **Kekurangan algoritma**: Sensitif terhadap noise dan outlier.

### Hyperparameter Tuning

Hyperparameter tuning dilakukan untuk menentukan konfigurasi terbaik yang dapat digunakan pada masing-masing model. Sehingga model dapat dilatih seoptimal mungkin agar memberikan hasil yang terbaik. Berikut merupakan contoh tuning pada model Random Forest:

```
param_grid_randomForest = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': (5, 10, 15, 20),
    'min_samples_leaf': (2, 5, 10, 15) ,
    'max_features': ['log2', 'sqrt',None],
}

random_forest = RandomForestRegressor()
grid_search_rf = GridSearchCV(random_forest, param_grid_randomForest, cv=5, scoring='neg_mean_squared_error', error_score='raise')
grid_search_rf.fit(X_train_encoded, y_train)
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
```

Proses ini dilakukan pada setiap model dan menghasilkan parameter terbaik untuk setiap modelnya, dengan rincian sebagai berikut:

- **K-Neighbors Regressor**
  ```
  {'n_neighbors': 15, 'weights': 'uniform'}
  ```

- **Random Forest Regressor**:
  ```
  {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 100}
  ```

- **AdaBoost Regressor**: 
  ```
  {'learning_rate': 0.01, 'n_estimators': 200}
  ```

### Pemilihan Model Terbaik

Setelah membandingkan performa model yang telah diterapkan Hyperparameter Tuning, model **Random Forest** memiliki nilai MSE paling rendah dibandingkan model lainnya. Nilai MSE terendah menunjukkan **model dengan kesalahan prediksi terendah**. Sehingga disimpulkan bahwa Random Forest adalah **model terbaik**, karena memberikan akurasi terbaik untuk melakukan prediksi nilai pada proyek ini.

## Evaluation

### Metrik Evaluasi

Metrik yang digunakan pada proyek ini adalah MSE atau Mean Squared Error. Metrik ini mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. Semakin kecil nilai MSE yang diperoleh, maka performa model akan semakin baik.

#### Formula dari MSE

```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```

Pada formula di atas, n adalah jumlah observasi, yᵢ adalah nilai aktual, dan ŷᵢ adalah nilai prediksi untuk setiap data ke-i. Prosesnya melibatkan pengurangan nilai prediksi dari nilai aktual, mengkuadratkan hasilnya untuk menghilangkan nilai negatif, lalu menjumlahkan semua kuadrat error dan membaginya dengan jumlah total data. Formula ini memberikan rata-rata error kuadrat yang mencerminkan akurasi model.

#### Cara Kerja dari MSE

MSE bekerja dengan menghitung kuadrat setiap error (perbedaan) untuk menghilangkan efek negatif dan memberi bobot lebih pada error besar, lalu mengambil rata-ratanya dari semua data. Oleh sebab itu, nilai MSE yang lebih kecil menunjukkan model yang lebih akurat dalam memprediksi nilai.

### Hasil Evaluasi Awal Sebelum Tuning

Evaluasi awal dari model yang masing-masing menggunakan parameter standar atau default diperoleh sebagai berikut, di mana **AdaBoost** memiliki nilai MSE yang lebih kecil dibandingkan model lainnya.

![Evaluation](https://raw.githubusercontent.com/rfqgal/submission-ml-terapan-1/refs/heads/master/images/evaluation.png)

Saat dilakukan 1 kali percobaan prediksi untuk nilai aktual 150.000, K-Neighbors Regressor memiliki nilai prediksi yang paling tinggi dengan rincian sebagai berikut:

2. KNN memprediksi 123.759.
3. AdaBoost memprediksi 121.699.
1. Random Forest memprediksi 121.306.

Namun tidak dapat disimpulkan bahwa model tersebut yang terbaik, karena percobaan prediksi baru dilakukan 1 kali saja.

### Hasil Evaluasi Setelah Dilakukannya Tuning

Berikutnya setelah dilakukan Hyperparameter Tuning, Random Forest memberikan hasil terbaik dengan MSE terendah, menunjukkan bahwa model ini mampu menjelaskan variansi data dengan baik dan memberikan prediksi yang akurat.

![Re-evaluation](https://raw.githubusercontent.com/rfqgal/submission-ml-terapan-1/refs/heads/master/images/re-evaluation.png)

Dalam tes prediksi untuk nilai aktual 150.000 diperoleh nilai prediksi sebagai berikut:

1. Random Forest memprediksi 117.165.
2. KNN memprediksi 118.083.
3. AdaBoost memprediksi 120.352.

Meskipun AdaBoost memberikan prediksi terdekat pada satu percobaan, **Random Forest tetap dipilih karena performa keseluruhan yang lebih baik berdasarkan metrik evaluasi**.
