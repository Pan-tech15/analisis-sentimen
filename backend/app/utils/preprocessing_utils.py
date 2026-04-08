import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Download stopwords NLTK jika belum
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Daftar stopword bahasa Indonesia dari NLTK + tambahan manual
stop_words = set(stopwords.words('indonesian'))
additional_stopwords = {'yg', 'nya', 'dgn', 'utk', 'pd', 'dg', 'dan', 'atau', 'juga', 'saja', 'pun', 'lah', 'kah', 'telah', 'sudah', 'akan', 'bisa', 'dapat', 'harus', 'ingin', 'jika', 'karena', 'sehingga', 'tetapi', 'namun', 'meskipun', 'walaupun', 'pada', 'dalam', 'ke', 'dari', 'dengan', 'untuk', 'bagi', 'oleh', 'sebagai', 'adalah', 'ini', 'itu', 'tersebut', 'mereka', 'kami', 'kita', 'anda', 'saya', 'kamu', 'dia'}
stop_words.update(additional_stopwords)

def case_folding(text):
    """Mengubah semua huruf menjadi lowercase"""
    return text.lower()

def normalize_text(text):
    """Normalisasi teks: menghapus tanda baca, angka, karakter khusus, dan mengganti slang (contoh sederhana)"""
    # Kamus slang sederhana (bisa diperluas)
    slang_dict = {
        'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'gk': 'tidak',
        'banget': 'sangat', 'bgt': 'sangat', 'dikit': 'sedikit', 'dikitt': 'sedikit',
        'jg': 'juga', 'jga': 'juga', 'msh': 'masih', 'udah': 'sudah',
        'udh': 'sudah', 'blm': 'belum', 'blom': 'belum', 'aj': 'saja',
        'aja': 'saja', 'sih': '', 'dong': '', 'deh': '', 'kok': '',
        'lho': '', 'yah': '', 'ya': '', 'wah': '', 'weh': ''
    }
    # Hapus mention, hashtag, URL
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Hapus angka dan tanda baca (kecuali spasi)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Ganti slang
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(words)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """Tokenisasi menggunakan NLTK word_tokenize"""
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    """Menghapus stopword dari token"""
    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    """Stemming menggunakan Sastrawi"""
    return [stemmer.stem(token) for token in tokens]

def clean_whitespace(text):
    """Membersihkan spasi ganda dan strip"""
    return ' '.join(text.split())

def preprocess_text(text, do_case_folding=True, do_normalize=True, do_stopword=True, do_stemming=True):
    """
    Pipeline preprocessing teks
    """
    if not isinstance(text, str):
        text = str(text)
    if do_case_folding:
        text = case_folding(text)
    if do_normalize:
        text = normalize_text(text)
    # Tokenisasi
    tokens = tokenize(text)
    if do_stopword:
        tokens = remove_stopwords(tokens)
    if do_stemming:
        tokens = stemming(tokens)
    # Gabung kembali
    result = ' '.join(tokens)
    result = clean_whitespace(result)
    return result