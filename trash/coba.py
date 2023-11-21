import streamlit as st
import pandas as pd
import numpy as np
from google_play_scraper import Sort, reviews, app
import re
import string  # Tambahkan baris ini
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from datetime import datetime

# Fungsi-fungsi preprocessing text yang ada pada file .ipynb
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # menghapus mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # menghapus hashtag
    text = re.sub(r'RT[\s]', '', text)  # menghapus RT
    text = re.sub(r"http\S+", '', text)  # menghapus link
    text = re.sub(r'[0-9]+', '', text)  # menghapus numbers
    text = text.replace('<br>', ' ')  # replace <br> ke dalam spasi
    text = text.replace('\n', ' ')  # replace baris baru ke dalam spasi
    text = text.translate(str.maketrans('', '', string.punctuation))  # Menghapus semua tanda baca
    text = text.strip(' ')  # hapus spasi karakter dari teks kiri dan kanan
    return text

def casefoldingText(text):  # Mengubah semua karakter dalam teks menjadi huruf kecil
    text = text.lower()
    return text

def tokenizingText(text):  # Tokenizing atau pemisahan string, teks menjadi daftar token
    text = word_tokenize(text)
    return text

def filteringText(text):  # Hapus stopwors dalam teks
    listStopwords = stopwords.words('indonesian')
    listStopwords.extend(["yg", "dg", "rt", "nya"])
    listStopwords = set(listStopwords)
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

def stemmingText(text):  # Pengurangan suatu kata menjadi kata dasar yang berimbuhan pada akhiran dan awalan atau pada akar kata
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def convertToSlangword(text):  # Merubah kata tidak baku menjadi kata baku
    kamusSlang = eval(open("slangwords.txt").read())
    pattern = re.compile(r'\b( ' + '|'.join(kamusSlang.keys()) + r')\b')
    content = []
    for kata in text:
        filterSlang = pattern.sub(lambda x: kamusSlang[x.group()], kata)
        content.append(filterSlang.lower())
    text = content
    return text

def toSentence(list_words):  # Ubah daftar kata menjadi kalimat
    sentence = ' '.join(word for word in list_words)
    return sentence

# Membuat fungsi untuk menggabungkan seluruh langkah text preprocessing
def text_preprocessing_process(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = tokenizingText(text)
    text = filteringText(text)
    text = stemmingText(text)
    text = convertToSlangword(text)
    return text

# Menambahkan kolom clean_teks pada tabel sorted_df_reviews
def add_clean_text_column(dataframe):
    dataframe['clean_teks'] = dataframe['content'].apply(text_preprocessing_process)
    return dataframe

# Main Streamlit app
st.title("TikTok Play Store Reviews")

# Input link aplikasi dari pengguna
app_link = "com.ss.android.ugc.trill"

# Tombol untuk memulai review dan informasi aplikasi
if st.button("Lihat Review Lama") or st.button("Lihat Review Terbaru"):
    # Implementasi fungsi reviews
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
    df_reviews = pd.DataFrame(result_reviews)
    sorted_df_reviews = add_clean_text_column(df_reviews)

    # Mendapatkan waktu sekarang
    current_time = datetime.now()

    # Menampilkan waktu di Streamlit
    st.write("Data Sentimen Analisis Review Aplikasi Tiktok diambil pada :", current_time)

    # DataFrame untuk hasil app
    df_app = pd.DataFrame.from_dict(result_app, orient='index', columns=['value'])
    st.write("Informasi Aplikasi:")
    st.dataframe(df_app)

    # Menampilkan hasil review
    st.write("Sorted Reviews:")
    if st.button("Simpan sebagai CSV"):
        sorted_df_reviews.to_csv('cobaReview.csv', index=False)
        st.success("DataFrame berhasil disimpan sebagai CSV.")
    st.dataframe(sorted_df_reviews)
