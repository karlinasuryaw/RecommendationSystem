# Laporan Sistem Rekomendasi - Karlina Surya Witanto

## Project Overview
Film merupakan sebuah hiburan yang banyak diminati oleh masyarakat dari berbagai kalangan, salah satunya anime movie. Banyaknya genre dari suatu film membuat
para penonton bingung untuk menentukan film apa yang ingin ditontonnya, seringkali mereka random menonton karena tidak ada informasi terkait film tersebut sebelumnya.
Tidak jarang seseorang yang ingin menonton film menjadi kebingungan karena terlalu banyak film yang tersedia di internet. Oleh karena itu, dibutuhkan sebuah 
sistem yang dapat membantu memberikan informasi yang sesuai dengan keinginan pengguna. Sistem tersebut sering disebut dengan sistem rekomendasi.
Sistem rekomendasi adalah suatu teknologi yang didesain untuk mempermudah pengguna dalam menemukan suatu data yang mungkin sesuai dengan profil pengguna
secara cepat dan dapat mengurangi jumlah informasi yang terlalu banyak [Vozalis & Margaritis, 2010](http://eprints.undip.ac.id/60611/1/laporan_24010312130054_1.pdf)

## Business Understanding
Masalah yang paling sering di hadapi para penonton yaitu kesulitan untuk memilih film yang ingin ditonton serta tidak adanya informasi yang mampu memberikan
rekomendasi terkait film yang ingin ditontonnya, selain itu para penonton keseringan tidak tahu film seperti apa yang disukainya dan tidak ada referensi khusus yang mampu membantunya. Oleh karena itu, saya tertarik untuk membuat suatu sistem rekomendasi dengan metode collaborative filtering. Dimana metode collaborative filtering ini mampu memprediksi apa yang akan disukai pengguna berdasarkan kesamaan mereka dengan pengguna lain. Selain itu metode ini juga mampu secara akurat merekomendasikan item kompleks seperti film tanpa memerlukan "pemahaman" dari item itu sendiri. Faktor yang digunakan dalam pengukuran (kesamaan pengguna atau kesamaan item) dalam sistem rekomendasi.

### Problem Statements
- Bagaimana penerapan metode collaborative filtering untuk sistem rekomendasi film anime?
- Bagimana hasil evaluasi dari performa metode collaborative filtering dengan metrik evaluasi?

### Goals
- Dapat mengimplementasikan metode collaborative filtering untuk sistem rekomendasi film anime
- Mengevaluasi performa dari metode collaborative filtering dengan metrik evaluasi.

### Solution statements
- **Collaborative Filtering**. Collaborative filtering, atau yang biasa disebut dengan crowd-wisdom adalah salah satu metode rekomendasi yang menggunakan data rating dari seorang pengguna, dan pengguna lain untuk menghasilkan rekomendasi. Collaborative filtering menganggap bahwa selera pengguna terhadap suatu item atau barang akan cenderung sama dari waktu ke waktu. Ditambah lagi, pengguna yang menyukai suatu item biasanya juga akan menyukai item lain yang disukai oleh pengguna lain yang juga menyukai item yang sama dengan pengguna tersebut.
Contoh gampangnya seperti ini, misalnya kalian menyukai film Lord of The Ring, kemudian menggunakan sistem rekomendasi, kalian ingin mengetahui film-film lainnya yang mirip atau mempunyai genre yang sama dengan Lord of The Ring. Dalam pemrosesan, sistem kemudian menemukan bahwa orang-orang yang menyukai Lord of The Ring biasanya juga suka film-film The Hobbit, Game of Thrones, dan Seven Kingdoms. Dari hasil tersebut, maka sistem memutuskan bahwa ketiga film itulah yang akan direkomendasikan kepada kalian.
[Materi Collaborative Filtering](https://www.twoh.co/2013/06/04/membuat-sistem-rekomendasi-menggunakan-item-based-collaborative-filtering/)

![Collaborative Filtering](https://dataconomy.com/wp-content/uploads/2015/03/Beginners-Guide-Recommender-Systems-Collaborative-Filtering-620x340.jpg)


## Data Understanding
Dataset yang digunakan merupakan [data rekomendasi anime](https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime/data) Dataset ini berisi informasi tentang 17.562 anime dan preferensi dari 325.772 pengguna yang berbeda dan terdiri dari empat file csv.
Content yang disediakan dari data ini yaitu :
* `animelist.csv` memiliki daftar semua anime yang didaftarkan oleh pengguna dengan skor masing-masing, status menonton, dan jumlah episode yang ditonton. Dataset ini berisi 109 Juta baris, 17.562 anime berbeda dan 325.772 pengguna berbeda.
    - `user_id`: ID Pengguna secara acak
    - `anime_id`: MyAnemlist ID pada anime.
    - `score`: skor antara 1 sampai 10 yang diberikan oleh pengguna. 0 jika pengguna tidak menetapkan    skor.
    - `watching_status`: ID negara dari anime pada daftar anime pengguna.
    - `watched_episodes`: jumlah episode yang ditonton oleh pengguna.
* `watching_status.csv` memaparkan setiap kemungkinan status kolom: "watching_status" pada file `animelist.csv`
* `anime.csv` berisi informasi umum dari setiap anime (17.562 anime berbeda) seperti genre, stats, studio, dll. Di bawah ini merupakan data yang digunakan pada `anime.csv` :
    - `MAL_ID` : ID Anime
    - `Name`   : nama anime
    - `eng_title` : Judul anime dalam Bahasa Inggris
    - `English name` : Nama anime dalam Bahasa Inggris
    - `Score` : nilai rata-rata yang diberikan oleh user
    - `Genres` : genre anime
    - `Episodes` : banyak episode yang ditonton oleh user
    - `Type` : TV, movie, OVA, etc.
    - `Premiered` : season premiere. (e.g. Spring 1998)
* `anime_with_synopsis.csv` berisi sinopsis semua anime. 

## Data Preparation
Proses untuk mempersiapkan data terdiri dari beberapa proses sebagai berikut :
- Melakukan proses filtering untuk mendiskualifikasi baris yang memiliki nilai unknown dimana proses ini bertujuan untuk menghapus informasi yang tidak relevan, kemudian dataframe diurutkan berdasarkan kolom score secara descending dengan code di bawah ini.
```
new_animedf = new_animedf[["MAL_ID", "Name", "Score", "Genres", "Episodes", "Type", "Premiered"]]
new_animedf = new_animedf[new_animedf["Score"] != "Unknown"]
new_animedf = new_animedf[new_animedf["Premiered"] != "Unknown"]
new_animedf = new_animedf[new_animedf["Type"] != "Unknown"]
new_animedf = new_animedf[new_animedf["Episodes"] != "Unknown"]
new_animedf = new_animedf[new_animedf["Genres"] != "Unknown"]
#mengurutkan dataframe berdasarkan kolom score
new_animedf.sort_values(by=['Score'], inplace=True,
                      ascending=False, kind='quicksort',
                      na_position='last')

#memilih beberapa kolom dataframe yang akan digunakan
new_animedf.tail(5)
```
- Removing missing value, tahapan ini diperlukan karena dengan tidak adanya missing value akan membuat performa dalam pembuatan model menjadi lebih baik. Tahapan ini dilakukan dengan code di bawah ini:
```
watch_sta = watch_sta.dropna()
anime_rating = anime_rating.dropna()
anime_synopsis = anime_synopsis.dropna()
```
- Normalisasi yaitu untuk mengubah nilai kolom numerik dalam kumpulan data ke skala umum, tanpa mendistorsi perbedaan dalam rentang nilai. Proses normalisasi dilakukan dengan metode Min Max, dengan rumus berikut :
![rumusminmax](https://user-images.githubusercontent.com/64744593/138431557-413c6bf0-66fc-4f73-ae9e-795fa5e574e4.png)

Proses tersebut dilakukan dengan code seperti di bawah ini :
```
anime_rating['rating'] = anime_rating["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values.astype(np.float32)
```
- Splitting data yaitu memisahkan data untuk proses latih dan proses untuk menguji kemampuan model yang akan di validasi mengunakan metrik evaluasi.

Berikut ini tahapan yang akan saya lakukan, yaitu :
- Tugas 1: Menemukan pengguna serupa (anime yang pernah di rate oleh pengguna)
- Tugas 2: Merekomendasikan Anime untuk pengguna acak

## Modeling
Pada pemodelan kali ini, saya menggunakan teknik embedding. Model Neural Collaborative Filtering (NCF) adalah jaringan saraf yang menyediakan penyaringan kolaboratif berdasarkan umpan balik implisit. Secara khusus, ini memberikan rekomendasi produk berdasarkan interaksi pengguna dan item. Data pelatihan untuk model ini harus berisi urutan pasangan (ID pengguna, ID anime) yang menunjukkan bahwa pengguna yang ditentukan telah berinteraksi dengan item, misalnya, dengan memberi peringkat atau mengklik. NCF pertama kali dijelaskan oleh Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu dan Tat-Seng Chua dalam makalah [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031).

Berikut ini merupakan tahapan yang saya lakukan, yaitu : 
1. Membuat kelas `GanbatteRec` yang didalamnya terdapat proses menghitung skor kecocokan antara user dan anime dengan teknik embedding.
2. Menginisialisasi model. Menggunakan layer `Dot` yang digunakan untuk komputasi antara embedding dari anime dan dari user.
3. Compile model tersebut dengan menggunakan loss binary_crossentropy, metrik MSE, metrik precission, dan metrik recall, serta menerapkan optimizernya Adam.
4. Menerapkan ModelCheckpoint() yang digunakan untuk menyimpan model terbaik dari proses training model.
5. Melakukan proses training menggunakan model.fit() dengan parameternya yakni batch_size = 5000, dan epoch 50.

Langkah untuk mendapatkan list rekomendasi anime berdasarkan aktivitas tontonan anime yang ditonton oleh user berdasarkan rate yang diberikan oleh user dengan batasan rata-rata ratenya sebesar 8.7:
1. Mencari data anime apa saja yang telah ditonton oleh user dan memasukkannya kedalam dataframe baru
2. Mencari nilai terendah dari rating anime
3. Membuat referensi anime (top_anime_refference) dengan mengurutkan anime berdasarkan rating.
4. Membuat dataframe baru berdasarkan dataframe utama (new_animedf) dan melakukan seleksi yang mana data yang dimasukkan adalah anime yang termasuk kedalam referensi anime (top_anime_refference)
5. Menghitung rata-rata rating yang diberikan oleh user
6. Membuat plot dari genre anime yang berada dalam dataframe baru

```
def get_user_anime_preference(user_id, plot=False, temp=1):
  already_watched_by_user = anime_rating[anime_rating.user_id==user_id]
  #rating_percentile menentukan batas rating terendah anime
  lowest_rating = np.percentile(already_watched_by_user.rating, 75)
  already_watched_by_user = already_watched_by_user[already_watched_by_user.rating >= lowest_rating]
  top_anime_refference = (
      already_watched_by_user.sort_values(by="rating", ascending=False)#.head(10)
      .anime_id.values
  )
  
  user_pref_df = new_animedf[new_animedf["MAL_ID"].isin(top_anime_refference)]
  user_pref_df = user_pref_df[["MAL_ID","Name", "Genres","Score","Episodes","Premiered"]]
  
  if temp != 0:
      print("User #{} Already Rated {} movies with average rating = {:.1f}".format(
        user_id, len(already_watched_by_user),
        already_watched_by_user['rating'].mean()*10,
      ))
  
      print('Recommended anime genre for user:')
  if plot:
      FavGenre(user_pref_df, plot)       
  return user_pref_df
```

Hasil dari rekomendasi dapat dilihat pada gambar dibawah ini :

Sebanyak 4821 user sudah melakukan rate terhadap 79 anime dengan rata-rata rating = 8.7

![task1](https://user-images.githubusercontent.com/64744593/138442579-e7ca71e5-9b1b-4f82-9b46-3c22e016342f.jpg)

Langkah untuk mendapatkan list rekomendasi anime untuk user secara acak :
1. Melakukan perulangan for untuk mencari preferensi dari anime yang telah ditonton oleh masing-masing user dan disimpan didalam list.
2. Membuat dataframe baru dengan data dari list anime dari masing-masing user dan sekaligus menghitung tingkat kemunculan dari judul anime tersebut.
3. Membuat perulangan for untuk mendapatkan data dari masing-masing anime, seperti genre, judul, jumlah episode

```
def get_anime_by_similar_users(similar_users):
  n=10
  rec_anime = []
  anime_list = []

  for user_id in similar_users.similar_users.values:
      pref_list = get_user_anime_preference(int(user_id), temp=0)
      pref_list = pref_list[~ pref_list.Name.isin(reff_user.Name.values)]
      anime_list.append(pref_list.Name.values)
      
  anime_list = pd.DataFrame(anime_list)
  sorted_list = pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts()).head(n)

  for i, anime_name in enumerate(sorted_list.index):        
      n_user_pref = sorted_list[sorted_list.index == anime_name].values[0][0]
      if isinstance(anime_name, str):
          
        frame = getanimedata(anime_name)
        anime_id = frame.MAL_ID.values[0]
        genre = frame.Genres.values[0]
        synopsis = getanimesynopsis(int(anime_id))
        rec_anime.append({"anime_id": anime_id ,
                                    "point": n_user_pref,
                                    "Title": anime_name, 
                                    "Genres": genre,
                                    "Sypnopsis": synopsis
                                    })


  return pd.DataFrame(rec_anime)
```
Hasil dari rekomendasi dapat dilihat pada gambar dibawah ini :

Di bawah ini merupakan hasil dari rekomendasi top 10 anime melalui model dari metode Collaborative Filtering yang sudah dibangun.

![task2](https://user-images.githubusercontent.com/64744593/138444328-4f581019-d290-477e-af56-4aa2fcb10f7c.jpg)


Ini adalah beberapa anime yang berada di daftar teratas berdasarkan genre yang direkomendasikan pada platform MyAnimeList yang telah diberikan rating berdasarkan user. Dan jika kita membandingkan hasil rekomendasi dari model yang telah saya bangun dengan genre anime teratas di MyAnimeList maka bisa disimpulkan bahwa akurasi model tersebut cukup tinggi secara empiris. Sebagian besar anime dalam daftar rekomendasi muncul pada daftar teratas dari beberapa genre di MyAnimeList ini. [Genre Action.](https://myanimelist.net/anime/genre/1/Action) [Genre Mysteri.](https://myanimelist.net/anime/genre/7/Mystery) [Genre Slice of Life.](https://myanimelist.net/anime/genre/36/Slice_of_Life)

![action](https://user-images.githubusercontent.com/64744593/138450574-2bdd8c90-39d9-4548-ac94-d72fefe2c628.jpeg)
![mysteri](https://user-images.githubusercontent.com/64744593/138450581-6827fb2e-c1d4-443d-b2d6-67e599973651.jpeg)
![slice](https://user-images.githubusercontent.com/64744593/138450582-3ef73ffa-be96-4938-9439-24accf756f79.jpeg)


## Evaluation
**Mean Squared Error (MSE)**. Teknik ini menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi. Pada tahap ini, jika nilai errornya semakin besar maka semakin besar nilai MSE yang dihasilkan.
Pada proyek ini nilai MSE yang didapatkan yaitu sebesar 0.0875

![Rumus MSE](https://user-images.githubusercontent.com/64744593/138072654-85f9754a-59ba-47d7-aaf2-9d71a0c3acd5.jpg)

**Precision, Recall, F-Measure**
- Precision adalah tingkat ketepatan antara informasi yang diminta oleh pengguna dengan jawaban yang diberikan oleh sistem. 
- Nilai Precision yang didapatkan pada proyek ini yaitu sebesar 0.8695

![rumusprecision](https://user-images.githubusercontent.com/64744593/138073031-aaa63f66-ed78-45cd-893e-1e6c3b4956e4.png)

- Recall adalah tingkat keberhasilan sistem dalam menemukan kembali sebuah informasi.
- Nilai recall yang didapatkan pada proyek ini yaitu sebesar 0.6572

![rumusrecall](https://user-images.githubusercontent.com/64744593/138073273-81ad2a6f-6559-4af9-8a37-f94bc11ab58b.png)

- F-Measure merupakan salah satu perhitungan evalusasi dalam informasi temu kembali yang mengkombinasikan recall dan precission. Nilai recall  dan Precission pada suatu keadaan dapat memiliki bobot yang berbeda. Ukuran yang menampilkan timbal balik antara Recall dan Precission adalah F-Measure yang merupakan bobot harmonic mean dan reall dan precission.
- Nilai F-Measure yang didapatkan pada proyek ini yaitu sebesar 0.7462

![rumusf1](https://user-images.githubusercontent.com/64744593/138073499-53730bd6-025b-4983-9c51-f5661c8ba58b.png)

Berikut ini merupakan tabel dari metrik evaluasi dari model diatas :

| Metrik    | Nilai  |
| --------- | ------ |
| MSE       | 0.0875 |
| Precision | 0.8695 |
| Recall    | 0.6572 |
| F-Measure | 0.7462 |
