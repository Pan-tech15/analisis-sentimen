import os
import sys
import re
import traceback
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from app import db
from app.models.training import Training

MODEL_FOLDER = 'data/models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# 7 emosi yang diizinkan (lowercase)
ALLOWED_EMOTIONS = {'senang', 'sedih', 'marah', 'takut', 'terkejut', 'percaya', 'netral'}

# ------------------ LOGGING & PROGRESS ------------------
def log(message, training_id=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f"[{timestamp}]"
    if training_id:
        prefix += f" [Training {training_id}]"
    print(f"{prefix} {message}", file=sys.stdout, flush=True)

def update_progress(app, training_id, progress, message=None):
    with app.app_context():
        training = Training.query.get(training_id)
        if training:
            training.progress = progress
            if message:
                if training.metrics is None:
                    training.metrics = {}
                training.metrics['progress_message'] = message
            db.session.commit()
            if message:
                log(message, training_id)

# ------------------ PREPROCESSING ------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------ BANGUN DICTIONARY LEXICON DARI DATA LATIH ------------------
def build_dictionary_lexicon(texts, labels, classes):
    """
    Membangun dictionary lexicon: kata -> dict(emosi -> 1/0)
    Jika kata pernah muncul di dokumen kelas tersebut, skor = 1, else 0.
    """
    lexicon = {c: {} for c in classes}
    for text, label in zip(texts, labels):
        words = set(text.split())
        for w in words:
            for c in classes:
                if c not in lexicon:
                    lexicon[c] = {}
                if w not in lexicon[c]:
                    lexicon[c][w] = 0
                if c == label:
                    lexicon[c][w] = 1  # set ke 1 jika pernah muncul di kelas ini
    return lexicon

def compute_dictionary_score(words, lexicon, classes):
    """Hitung skor dictionary untuk satu kalimat (jumlah kata yang ada di lexicon kelas)."""
    scores = np.zeros(len(classes))
    for w in words:
        for i, c in enumerate(classes):
            scores[i] += lexicon.get(c, {}).get(w, 0)
    return scores

# ------------------ PMI ------------------
def compute_class_word_freq(texts, labels):
    class_word_counts = {}
    class_doc_counts = {}
    for text, label in zip(texts, labels):
        words = text.split()
        class_doc_counts[label] = class_doc_counts.get(label, 0) + 1
        if label not in class_word_counts:
            class_word_counts[label] = {}
        for w in words:
            class_word_counts[label][w] = class_word_counts[label].get(w, 0) + 1
    return class_word_counts, class_doc_counts

def compute_pmi(word, emotion, class_word_counts, class_doc_counts, total_docs, vocab_size):
    joint = class_word_counts.get(emotion, {}).get(word, 0) + 1  # Laplace smoothing
    p_joint = joint / (total_docs + vocab_size)

    word_total = sum([class_word_counts[c].get(word, 0) for c in class_word_counts]) + 1
    p_word = word_total / (total_docs + vocab_size)

    p_emotion = (class_doc_counts.get(emotion, 0) + 1) / (total_docs + len(class_doc_counts))

    pmi = np.log(p_joint / (p_word * p_emotion))
    return max(0.0, pmi)

def compute_pmi_scores(words, classes, class_word_counts, class_doc_counts, total_docs, vocab_size):
    """Hitung skor PMI untuk satu kalimat (jumlah PMI setiap kata)."""
    scores = np.zeros(len(classes))
    for w in words:
        for i, c in enumerate(classes):
            scores[i] += compute_pmi(w, c, class_word_counts, class_doc_counts, total_docs, vocab_size)
    return scores

# ------------------ MAIN TRAINING FUNCTION ------------------
def train_lexicon_nb(app, training_id, config, dataset_path):
    with app.app_context():
        training = Training.query.get(training_id)
        if not training:
            log(f"Training {training_id} tidak ditemukan", training_id)
            return
        training.status = 'running'
        db.session.commit()
        log(f"Memulai training Lexicon-NB dengan konfigurasi: {config.name}", training_id)

        try:
            # 1. Baca dataset
            log(f"Membaca dataset dari {dataset_path}", training_id)
            df = pd.read_csv(dataset_path)
            log(f"Dataset loaded: {len(df)} baris", training_id)

            if 'kalimat' not in df.columns or 'emotion' not in df.columns:
                raise ValueError("Dataset harus memiliki kolom 'kalimat' dan 'emotion'")

            # Bersihkan label (hanya 7 emosi valid)
            def clean_label(val):
                s = str(val).strip().lower()
                return s if s in ALLOWED_EMOTIONS else None

            df['emotion_clean'] = df['emotion'].apply(clean_label)
            df = df.dropna(subset=['emotion_clean']).reset_index(drop=True)

            if len(df) == 0:
                raise ValueError("Dataset kosong setelah membersihkan label tidak valid.")

            update_progress(app, training_id, 5, "Memulai preprocessing teks...")
            texts = [preprocess_text(t) for t in df['kalimat']]
            labels = df['emotion_clean'].tolist()
            total_samples = len(texts)
            log(f"Jumlah sampel setelah validasi: {total_samples}", training_id)
            log(f"Label unik: {set(labels)}", training_id)

            # 2. Parameter dari config
            params = config.params
            general = params.get('general', {})
            split_cfg = params.get('split', {})
            nb_params = params.get('naivebayes', {})
            fusion_params = params.get('fusion', {})

            random_state = int(general.get('randomState', 42))
            shuffle = general.get('shuffle', 'yes') == 'yes'
            stratified = general.get('stratified', 'yes') == 'yes'
            split_type = split_cfg.get('type', 'percentage')
            test_ratio = float(split_cfg.get('test', 20)) / 100

            model_type = nb_params.get('modelType', 'MultinomialNB')
            feature_type = nb_params.get('feature', 'TfidfVectorizer')
            alpha = float(nb_params.get('alpha', 1.0))
            fit_prior = nb_params.get('fitPrior', 'True') == 'True'

            update_progress(app, training_id, 10, "Membuat vectorizer...")
            if feature_type == 'CountVectorizer':
                vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,2))
            else:
                vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

            log("Melakukan fit vectorizer...", training_id)
            X_text = vectorizer.fit_transform(texts)
            log(f"Shape fitur: {X_text.shape}", training_id)

            # 4. Label encoding
            le = LabelEncoder()
            y = le.fit_transform(labels)
            classes = le.classes_.tolist()
            log(f"Label encoded: {classes}", training_id)

            update_progress(app, training_id, 25, "Membangun dictionary lexicon & PMI...")

            # 5. Bangun dictionary lexicon dari data latih
            dict_lexicon = build_dictionary_lexicon(texts, labels, classes)
            log(f"Dictionary lexicon dibangun untuk {len(classes)} kelas", training_id)

            # 6. Hitung frekuensi untuk PMI
            class_word_counts, class_doc_counts = compute_class_word_freq(texts, labels)
            total_docs = len(texts)
            vocab = set()
            for t in texts:
                vocab.update(t.split())
            vocab_size = len(vocab)

            # 7. Hitung lexicon scores (Dictionary + PMI) per sampel
            lexicon_scores = []
            log("Menghitung skor lexicon (Dictionary + PMI)...", training_id)
            start = time.time()
            for i, text in enumerate(texts):
                words = text.split()
                # Skor dictionary
                dict_scores = compute_dictionary_score(words, dict_lexicon, classes)
                # Skor PMI
                pmi_scores = compute_pmi_scores(words, classes, class_word_counts, class_doc_counts, total_docs, vocab_size)
                # Gabungan
                combined = dict_scores + pmi_scores
                lexicon_scores.append(combined)

                if (i+1) % 500 == 0 or i+1 == total_samples:
                    progress = 25 + int(((i+1)/total_samples)*15)
                    msg = f"Lexicon scoring: {i+1}/{total_samples}"
                    log(msg, training_id)
                    update_progress(app, training_id, progress, msg)

            lexicon_scores = np.array(lexicon_scores)
            log(f"Lexicon scores shape: {lexicon_scores.shape}", training_id)

            # 8. Inisialisasi model Naive Bayes
            if model_type == 'BernoulliNB':
                model = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
            else:
                model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

            update_progress(app, training_id, 45, "Memulai pelatihan model...")

            # 9. Split data
            X_train, X_test, y_train, y_test, lex_train, lex_test = train_test_split(
                X_text, y, lexicon_scores,
                test_size=test_ratio,
                random_state=random_state,
                shuffle=shuffle,
                stratify=y if stratified and len(np.unique(y)) > 1 else None
            )
            log(f"Data split: train={X_train.shape[0]}, test={X_test.shape[0]}", training_id)

            # 10. Training NB
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            log("Model NB selesai dilatih", training_id)

            update_progress(app, training_id, 65, "Melakukan fusi dengan lexicon...")

            # 11. Fusion
            method = fusion_params.get('method', 'product')
            if method == 'sum':
                final_scores = proba + lex_test
            elif method == 'weighted':
                w = float(fusion_params.get('weight', 0.5))
                final_scores = w * proba + (1 - w) * lex_test
            else:  # product
                # Normalisasi lexicon_test ke rentang positif
                lex_test_norm = (lex_test - lex_test.min(axis=1, keepdims=True)) / (lex_test.max(axis=1, keepdims=True) - lex_test.min(axis=1, keepdims=True) + 1e-8)
                final_scores = proba * (1 + lex_test_norm)

            # Softmax normalisasi
            exp_scores = np.exp(final_scores - np.max(final_scores, axis=1, keepdims=True))
            final_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            y_pred = np.argmax(final_scores, axis=1)

            # 12. Evaluasi
            update_progress(app, training_id, 80, "Menghitung metrik evaluasi...")
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()

            metrics = {
                'accuracy': round(acc, 4),
                'f1_score': round(f1, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'confusion_matrix': cm,
                'class_labels': classes
            }
            log(f"Evaluasi selesai. Akurasi: {metrics['accuracy']}, F1: {metrics['f1_score']}", training_id)

            # 13. Simpan model
            model_filename = f"lexicon_nb_{training_id}.pkl"
            model_path = os.path.join(MODEL_FOLDER, model_filename)
            artifacts = {
                'model': model,
                'vectorizer': vectorizer,
                'label_encoder': le,
                'dict_lexicon': dict_lexicon,
                'classes': classes,
                'config': params
            }
            joblib.dump(artifacts, model_path)
            log(f"Model disimpan di {model_path}", training_id)

            training.status = 'completed'
            training.progress = 100
            training.metrics = metrics
            training.model_path = model_path
            training.completed_at = datetime.utcnow()
            db.session.commit()
            log(f"Training {training_id} selesai!", training_id)

        except Exception as e:
            error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
            log(error_msg, training_id)
            training.status = 'failed'
            training.metrics = {'error': str(e), 'progress_message': f'Gagal: {str(e)}'}
            training.progress = 0
            db.session.commit()
            raise e