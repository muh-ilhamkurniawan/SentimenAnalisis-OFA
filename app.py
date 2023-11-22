import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np 
from google_play_scraper import Sort, reviews, app  
import matplotlib.pyplot as plt
from datetime import datetime
from function import text_preprocessing_process, sentiment_analysis_lexicon_indonesia, generate_wordcloud
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import os
import csv

#navigasi sidebar
with st.sidebar:
    selected = option_menu("Sentimen Analisis di Playstore", 
                           ["Info TikTok","Cek Review", "Cek Aplikasi Lain"], 
        icons=['house', 'gear', 'gear'], 
        menu_icon="cast", default_index=0)
    selected
if (selected == "Info TikTok"):
    # Main Streamlit app
    st.title("TikTok Play Store Reviews")
    # Input link aplikasi dari pengguna
    app_link = "com.ss.android.ugc.trill"
    # Tombol untuk memulai review dan informasi aplikasi

    # Membuat beberapa tombol secara horizontal
    button1, button2= st.columns(2)

    if button1.button("Lihat Review Lama"):
        def baca_csv(file_path, encoding='utf-8'):
            with open(file_path, 'r', encoding=encoding) as file:
                csv_reader = csv.reader(file)
                # Membaca header
                header = next(csv_reader)
                
                # Membaca baris-baris berikutnya
                rows = list(csv_reader)

            # Membuat DataFrame dengan menggunakan header sebagai nama kolom
            df_info = pd.DataFrame(rows, columns=header)
            
            return df_info

        # Implementasi lihat reviews lama
        file_path1 = './static/InfoTiktok.csv'
        file_path2 = './static/TiktokReview.csv'
        df_info = baca_csv(file_path1)
        df_reviews = baca_csv(file_path2)

        # Menampilkan waktu di Streamlit
        current_time = df_info[['value']].iloc[-1].values[0]
        st.write("Data Sentimen Analisis Review Aplikasi Tiktok diambil pada :", current_time)
                

        # Menampilkan hasil review
        st.subheader("Informasi Aplikasi:")
        st.dataframe(df_info)

        # Menampilkan hasil review
        st.subheader("Sorted Reviews:")
        st.dataframe(df_reviews[['userName', 'score', 'at', 'content']])

        # Menampilkan hasil review setelah preprocessing dan analisis sentimen
        st.subheader("Sorted Reviews after Text Preprocessing and Sentiment Analysis:")
        st.dataframe(df_reviews[['userName', 'score', 'at', 'content', 'clean_teks', 'polarity_score', 'polarity']])

        # Tampilkan WordCloud dari kolom 'clean_teks'
        if 'clean_teks' in df_reviews.columns:
            clean_text_combined = ' '.join(df_reviews['clean_teks'])
            wordcloud = generate_wordcloud(clean_text_combined)

            # Tampilkan WordCloud menggunakan Matplotlib
            st.subheader("WordCloud Hasil:")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Kolom 'clean_teks' tidak ditemukan dalam DataFrame.")


        # Hitung jumlah masing-masing nilai di kolom 'polarity'
        polarity_counts = df_reviews['polarity'].value_counts()

        # Fungsi untuk membuat pie chart
        def create_pie_chart(polarity_counts):
            fig, ax = plt.subplots()
            ax.pie(polarity_counts, labels=polarity_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Memastikan pie chart berbentuk lingkaran
            return fig

        # Main Streamlit app
        st.subheader("Pie Chart of Polarity Distribution")

        # Menampilkan pie chart
        st.pyplot(create_pie_chart(polarity_counts))

    if button2.button("Lihat Review Terbaru"):
        # Implementasi fungsi reviews baru
        result_reviews, continuation_token = reviews(
            app_link,
            lang='id',
            country='id',
            sort=Sort.MOST_RELEVANT,
            count=100,
            filter_score_with=None
        )

        # Implementasi fungsi app
        result_app = app(
            app_link,
            lang='id',
            country='id',
        )

        # DataFrame untuk hasil reviews
        df_reviews = pd.DataFrame(np.array(result_reviews), columns=['review'])
        df_reviews = df_reviews.join(pd.DataFrame(df_reviews.pop('review').tolist()))
        sorted_df_reviews = df_reviews[['userName', 'score', 'at', 'content']]

        # Mendapatkan waktu sekarang
        current_time = datetime.now()

        # Menampilkan waktu di Streamlit
        st.write("Data Sentimen Analisis Review Aplikasi Tiktok diambil pada :", current_time)

        # DataFrame untuk hasil app
        df_app = pd.DataFrame.from_dict(result_app, orient='index', columns=['value'])
        st.write("Informasi Aplikasi:")
        st.dataframe(df_app)

        new_row = pd.Series({'value': current_time}, name='time')
        # Menambahkan baris ke DataFrame menggunakan loc
        df_app.loc['time'] = new_row

        df_app.to_csv('InfoTikTok.csv', index_label='key')

        # Menampilkan hasil review
        st.write("Sorted Reviews:")
        st.dataframe(sorted_df_reviews)

        # Menambahkan kolom clean_teks pada tabel sorted_df_reviews
        sorted_df_reviews['clean_teks'] = sorted_df_reviews['content'].apply(text_preprocessing_process)

        # Hasil dari penentuan polaritas sentimen tweet
        results = sorted_df_reviews['clean_teks'].apply(sentiment_analysis_lexicon_indonesia)
        results = list(zip(*results))
        sorted_df_reviews['polarity_score'] = results[0]
        sorted_df_reviews['polarity'] = results[1]

        # Menampilkan hasil review setelah preprocessing dan analisis sentimen
        st.write("Sorted Reviews after Text Preprocessing and Sentiment Analysis:")
        sorted_df_reviews.to_csv('TikTokReview.csv', index=False)
        st.dataframe(sorted_df_reviews[['userName', 'score', 'at', 'content', 'clean_teks', 'polarity_score', 'polarity']])

        # Hitung jumlah masing-masing nilai di kolom 'polarity'
        polarity_counts = sorted_df_reviews['polarity'].value_counts()

        # Fungsi untuk membuat pie chart
        def create_pie_chart(polarity_counts):
            fig, ax = plt.subplots()
            ax.pie(polarity_counts, labels=polarity_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Memastikan pie chart berbentuk lingkaran
            return fig

        # Main Streamlit app
        st.write("Pie Chart of Polarity Distribution")

        # Menampilkan pie chart
        st.pyplot(create_pie_chart(polarity_counts))

if (selected == "Cek Review"):
    # Load model yang sudah di-export
    model = load_model('model.h5')
    st.title ('Prediksi Ulasan')
    # Contoh teks baru yang ingin diprediksi
    new_text = st.text_area("Masukkan teks yang ingin diprediksi:")

    # Tombol untuk memulai prediksi
    if st.button("Prediksi"):
        # Memeriksa apakah input kosong
        if not new_text:
            st.error("Ulasan diperlukan. Silakan masukkan ulasan yang ingin diprediksi")
        else:
            # Load data yang digunakan saat melatih model (sebagai contoh)
            # Gantilah path dan nama file dengan data yang Anda gunakan saat melatih model
            train_data = pd.read_csv('ML2.csv')

            # Preprocessing teks baru
            # Tokenisasi
            tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
            tokenizer.fit_on_texts(train_data['content'])  # Menggunakan teks yang digunakan saat melatih model

            # Sequencing dan padding
            new_text_seq = tokenizer.texts_to_sequences([new_text])
            max_len = 100  # Sesuaikan dengan panjang yang digunakan saat melatih model
            new_text_padded = pad_sequences(new_text_seq, maxlen=max_len, padding='post', truncating='post')

            # Prediksi probabilitas untuk setiap kelas
            predicted_probabilities = model.predict(new_text_padded)[0]

            # Mengambil indeks kelas dengan probabilitas tertinggi
            predicted_label = np.argmax(predicted_probabilities)

            # Mengubah label numerik menjadi label asli
            label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            predicted_class = label_mapping[predicted_label]

            # Menampilkan hasil prediksi
            st.write(f'Prediksi Sentimen: {predicted_class}')

if (selected == "Cek Aplikasi Lain"):
    # Main Streamlit app
    st.title("Play Store Reviews Apk")

    # Input link aplikasi dari pengguna
    app_link = st.text_input("Masukkan link aplikasi")
    app_number = st.slider("Masukkan jumlah", 1, 100)
    def extract_package_name(url):
        pattern = r'id=([^\&]+)'
        match = re.search(pattern, url)
        
        if match:
            return match.group(1)

        return None
    app_link = extract_package_name(app_link)
    # Tombol untuk memulai review dan informasi aplikasi
    if st.button("Mulai Review"):
        # Memeriksa apakah input kosong
        if not app_link:
            st.error("Link Aplikasi diperlukan. Silakan masukkan link aplikasi.")
        else:
            # Implementasi fungsi reviews
            result_reviews, continuation_token = reviews(
                app_link,
                lang='id',
                country='id',
                sort=Sort.MOST_RELEVANT,
                count=app_number,
                filter_score_with=None
            )

            # Implementasi fungsi app
            result_app = app(
                app_link,
                lang='id',
                country='id',
            )

            # DataFrame untuk hasil reviews
            df_reviews = pd.DataFrame(np.array(result_reviews), columns=['review'])
            df_reviews = df_reviews.join(pd.DataFrame(df_reviews.pop('review').tolist()))
            sorted_df_reviews = df_reviews[['userName', 'score', 'at', 'content']]

            # DataFrame untuk hasil app
            df_app = pd.DataFrame.from_dict(result_app, orient='index', columns=['value'])
            st.write("Informasi Aplikasi:")
            st.dataframe(df_app)

            # Menampilkan hasil review
            st.write("Sorted Reviews:")
            st.dataframe(sorted_df_reviews)

            # Menambahkan kolom clean_teks pada tabel sorted_df_reviews
            sorted_df_reviews['clean_teks'] = sorted_df_reviews['content'].apply(text_preprocessing_process)

            # Hasil dari penentuan polaritas sentimen tweet
            results = sorted_df_reviews['clean_teks'].apply(sentiment_analysis_lexicon_indonesia)
            results = list(zip(*results))
            sorted_df_reviews['polarity_score'] = results[0]
            sorted_df_reviews['polarity'] = results[1]

            # Menampilkan hasil review setelah preprocessing dan analisis sentimen
            st.write("Sorted Reviews after Text Preprocessing and Sentiment Analysis:")
            st.dataframe(sorted_df_reviews[['userName', 'score', 'at', 'content', 'clean_teks', 'polarity_score', 'polarity']])

            # Hitung jumlah masing-masing nilai di kolom 'polarity'
            polarity_counts = sorted_df_reviews['polarity'].value_counts()

            # Fungsi untuk membuat pie chart
            def create_pie_chart(polarity_counts):
                fig, ax = plt.subplots()
                ax.pie(polarity_counts, labels=polarity_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Memastikan pie chart berbentuk lingkaran
                return fig

            # Main Streamlit app
            st.write("Pie Chart of Polarity Distribution")

            # Menampilkan pie chart
            st.pyplot(create_pie_chart(polarity_counts))
