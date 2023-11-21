import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model yang sudah di-export
model = load_model('model.h5')
st.title ('Prediksi Ulasan')
# Contoh teks baru yang ingin diprediksi
new_text = st.text_area("Masukkan teks yang ingin diprediksi:")

# Tombol untuk memulai prediksi
if st.button("Prediksi"):
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
