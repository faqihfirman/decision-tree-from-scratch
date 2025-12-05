🌳 Decision Tree ID3 From Scratch

Implementasi algoritma Decision Tree ID3 berbasis entropi secara from scratch menggunakan Python.
Pustaka ini dirancang sebagai model white-box dengan kemampuan visualisasi pohon dan decision path sehingga cocok untuk aplikasi Explainable AI (XAI), terutama pada Pertanian Presisi.

📁 Struktur Modul
1. modul_encoding.py

Modul untuk menangani encoding label kategorikal.

Fitur:
- Mengubah label string → numerik.
- Menjamin konsistensi mapping antara training dan testing.
- Digunakan pada semua skenario eksperimen.

2. modul_split_data.py

Modul pembagi dataset menggunakan Stratified K-Fold.

Fitur:
- Mempertahankan proporsi kelas di setiap fold.
- Mencegah bias evaluasi pada dataset imbalanced.
- Mendukung proses 5-Fold Cross Validation.

3. modul_decision_tree.py

Modul inti implementasi algoritma ID3.

Komponen Utama:
- Class Node
  - Menyimpan fitur pemisah, threshold, dan anak kiri/kanan
  - Mendukung UUID untuk identifikasi node

- Class DecisionTreeClassifier
  - Menghitung Entropi dan Information Gain
  - Memilih fitur terbaik secara greedy
  - Membentuk pohon secara rekursif
  - Mendukung stopping criteria:
      - max depth
      - kelas homogen
      - IG = 0 → leaf node

4. modul_visualization.py

Modul untuk menampilkan Explainable Output.

Fitur:
- Visualisasi struktur pohon keputusan.
- Visualisasi decision path untuk satu sampel input.
- Pewarnaan node untuk memudahkan interpretasi.
- Mode tampilan ringkas (pruned view).

5. modul_metrics.py
o
Modul evaluasi model secara manual tanpa library eksternal.

Menyediakan:
- Accuracy
- Precision
- Recall
- F1-Sc0re

Digunakan untuk membuat classification report pada kedua skenario.
