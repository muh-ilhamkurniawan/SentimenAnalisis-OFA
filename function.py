import re
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import os
# membuat functions untuk preprocessing text

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # menghapus mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # menghapus hashtag
    text = re.sub(r'RT[\s]', '', text) # menghapus RT
    text = re.sub(r"http\S+", '', text) # menghapus link
    text = re.sub(r'[0-9]+', '', text) # menghapus numbers

    text = text.replace('<br>', ' ') # replace <br> ke dalam spasi
    text = text.replace('\n', ' ') # replace baris baru ke dalam spasi
    text = text.translate(str.maketrans('', '', string.punctuation)) # Menghapus semua tanda baca
    text = text.strip(' ') # hapus spasi karakter dari teks kiri dan kanan
    return text

def casefoldingText(text): # Mengubah semua karakter dalam teks menjadi huruf kecil
    text = text.lower()
    return text

def tokenizingText(text): # Tokenizing atau pemisahan string, teks menjadi daftar token
    text = word_tokenize(text)
    return text

def filteringText(text): # Hapus stopwors dalam teks
    listStopwords = stopwords.words('indonesian')
    listStopwords.extend(["yg","dg","rt","nya",])
    listStopwords = set(listStopwords)
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

def stemmingText(text): # Pengurangan suatu kata menjadi kata dasar yang berimbuhan pada akhiran dan awalan atau pada akar kata
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def convertToSlangword(text): # Merubah kata tidak baku menjadi kata baku
    kamusSlang = eval(open("slangwords.txt").read())
    pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
    content = []
    for kata in text:
        filterSlang = pattern.sub(lambda x: kamusSlang[x.group()],kata)
        content.append(filterSlang.lower())
    text = content
    return text

def toSentence(list_words): # Ubah daftar kata menjadi kalimat
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

# Memuat data positif dan negatif leksikon
lexicon_positive = dict()

def read_lexicon_csv(file_path, key_column, value_column):
    lexicon_positive = dict()
    
    # Membaca file CSV ke dalam DataFrame
    df_info = pd.read_csv(file_path, sep=',')

    # Iterasi melalui setiap baris dalam DataFrame dan menambahkannya ke dalam kamus
    for index, row in df_info.iterrows():
        lexicon_positive[row[key_column]] = int(row[value_column])

    return lexicon_positive

# Contoh pemanggilan fungsi
lexicon_positive = read_lexicon_csv('lexicon_positive.csv', 'key', 'value')


lexicon_negative = dict()

def read_lexicon_csv(file_path, key_column, value_column):
    lexicon_negative = dict()
    
    # Membaca file CSV ke dalam DataFrame
    df_info = pd.read_csv(file_path, sep=',')

    # Iterasi melalui setiap baris dalam DataFrame dan menambahkannya ke dalam kamus
    for index, row in df_info.iterrows():
        lexicon_negative[row[key_column]] = int(row[value_column])

    return lexicon_negative

# Contoh pemanggilan fungsi
lexicon_negative = read_lexicon_csv('lexicon_negative.csv', 'key', 'value')

# Fungsi untuk menentukan polaritas sentimen tweet
def sentiment_analysis_lexicon_indonesia(text):
    # untuk kata dalam teks:
    score = 0
    for word in text:
        if word in lexicon_positive:
            score = score + lexicon_positive[word]
    for word in text:
        if word in lexicon_negative:
            score = score + lexicon_negative[word]
    polarity = ''
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return score, polarity

# Fungsi untuk membuat WordCloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud