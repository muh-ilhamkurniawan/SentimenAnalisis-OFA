import re
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

with open('lexicon_positive.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
with open('lexicon_negative.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])

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