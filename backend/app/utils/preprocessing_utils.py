import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Daftar stopword manual (lengkap)
stop_words = set([
    'yang', 'dan', 'atau', 'juga', 'saja', 'pun', 'lah', 'kah', 'telah', 'sudah',
    'akan', 'bisa', 'dapat', 'harus', 'ingin', 'jika', 'karena', 'sehingga',
    'tetapi', 'namun', 'meskipun', 'walaupun', 'pada', 'dalam', 'ke', 'dari',
    'dengan', 'untuk', 'bagi', 'oleh', 'sebagai', 'adalah', 'ini', 'itu',
    'tersebut', 'mereka', 'kami', 'kita', 'anda', 'saya', 'kamu', 'dia',
    'yg', 'nya', 'dgn', 'utk', 'pd', 'dg', 'gak', 'ga', 'nggak', 'gk',
    'banget', 'bgt', 'dikit', 'dikitt', 'jg', 'jga', 'msh', 'udah', 'udh',
    'blm', 'blom', 'aj', 'aja', 'sih', 'dong', 'deh', 'kok', 'lho', 'yah',
    'ya', 'wah', 'weh', 'sih', 'dong', 'deh', 'kok', 'lho'
])

def case_folding(text): #mengubah seluruh huruf ke lowercase
    return text.lower()

def normalize_text(text):
    # Hapus mention, hashtag, URL
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Hapus angka dan tanda baca (kecuali spasi)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Ganti slang
    slang_dict = {
        'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'gk': 'tidak',
        'banget': 'sangat', 'bgt': 'sangat', 'dikit': 'sedikit', 'dikitt': 'sedikit',
        'jg': 'juga', 'jga': 'juga', 'msh': 'masih', 'udah': 'sudah',
        'udh': 'sudah', 'blm': 'belum', 'blom': 'belum', 'aj': 'saja',
        'aja': 'saja', 'sih': '', 'dong': '', 'deh': '', 'kok': '',
        'lho': '', 'yah': '', 'ya': '', 'wah': '', 'weh': ''
    }
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text): #memecah teks bersih menjadi token-token per kata individu
    return text.split()

def remove_stopwords(tokens): #menghilangkan kata-kata umum
    return [token for token in tokens if token not in stop_words]

def stemming(tokens): #mengubah setiap kata menjadi bentuk kata dasar
    return [stemmer.stem(token) for token in tokens]

def clean_whitespace(text): #menghapus spasi ganda atau berlebihan
    return ' '.join(text.split())

def preprocess_text(text, do_case_folding=True, do_normalize=True, do_stopword=True, do_stemming=True):
    if not isinstance(text, str):
        text = str(text)
    if do_case_folding:
        text = case_folding(text)
    if do_normalize:
        text = normalize_text(text)
    tokens = tokenize(text)
    if do_stopword:
        tokens = remove_stopwords(tokens)
    if do_stemming:
        tokens = stemming(tokens)
    result = ' '.join(tokens)
    result = clean_whitespace(result)
    return result