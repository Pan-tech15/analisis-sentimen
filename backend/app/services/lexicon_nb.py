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
                    lexicon[c][w] = 1
    return lexicon

def compute_dictionary_score(words, lexicon, classes):
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
    joint = class_word_counts.get(emotion, {}).get(word, 0) + 1
    p_joint = joint / (total_docs + vocab_size)
    word_total = sum([class_word_counts[c].get(word, 0) for c in class_word_counts]) + 1
    p_word = word_total / (total_docs + vocab_size)
    p_emotion = (class_doc_counts.get(emotion, 0) + 1) / (total_docs + len(class_doc_counts))
    pmi = np.log(p_joint / (p_word * p_emotion))
    return max(0.0, pmi)

def compute_pmi_scores(words, classes, class_word_counts, class_doc_counts, total_docs, vocab_size):
    scores = np.zeros(len(classes))
    for w in words:
        for i, c in enumerate(classes):
            scores[i] += compute_pmi(w, c, class_word_counts, class_doc_counts, total_docs, vocab_size)
    return scores

# ------------------ FUNGSI TIE-BREAKER ------------------
def predict_with_tiebreaker(final_scores, classes, tiebreaker):
    n_samples = final_scores.shape[0]
    y_pred = np.zeros(n_samples, dtype=int)
    rank_array = np.array([tiebreaker.get(c, 99) for c in classes])
    for i in range(n_samples):
        max_val = np.max(final_scores[i])
        max_indices = np.where(final_scores[i] == max_val)[0]
        if len(max_indices) == 1:
            y_pred[i] = max_indices[0]
        else:
            ranks_tied = rank_array[max_indices]
            y_pred[i] = max_indices[np.argmin(ranks_tied)]
    return y_pred

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

            # ========== DUKUNGAN KELAS non_idiom ==========
            has_idiom_col = 'has_idiom' in df.columns

            def clean_and_map_emotion(raw_value):
                """Bersihkan dan petakan label ke 7 emosi standar. None jika tidak valid."""
                if pd.isna(raw_value):
                    return None
                s = str(raw_value).strip().lower()
                return s if s in ALLOWED_EMOTIONS else None

            def determine_label(row):
                # Jika kolom has_idiom tersedia, gunakan itu
                if has_idiom_col:
                    is_idiom = row['has_idiom']
                    if isinstance(is_idiom, str):
                        is_idiom = is_idiom.strip().lower() in ('ya', 'yes', 'true', '1')
                    else:
                        is_idiom = bool(is_idiom)
                    if not is_idiom:
                        return 'non_idiom'
                # Jika ada idiom (atau tidak ada kolom has_idiom), gunakan emosi
                emo = row['emotion']
                if pd.isna(emo):
                    return 'non_idiom'
                emo_clean = clean_and_map_emotion(emo)
                if emo_clean is None:
                    return 'non_idiom'
                return emo_clean

            df['label'] = df.apply(determine_label, axis=1)

            # Hapus data yang tidak memiliki label (seharusnya tidak ada karena penanganan di atas)
            before = len(df)
            df = df[df['label'].notna()].reset_index(drop=True)
            log(f"Data setelah pembersihan label: {len(df)} baris (dari {before})", training_id)

            if len(df) == 0:
                raise ValueError("Dataset kosong setelah membersihkan label.")

            # Distribusi label
            label_counts = df['label'].value_counts().to_dict()
            log(f"Distribusi label: {label_counts}", training_id)

            # 2. Parameter dari config
            params = config.params
            general = params.get('general', {})
            split_cfg = params.get('split', {})
            nb_params = params.get('naivebayes', {})
            fusion_params = params.get('fusion', {})

            random_state = int(general.get('randomState', 42))
            shuffle = general.get('shuffle', 'yes') == 'yes'
            stratified = general.get('stratified', 'yes') == 'yes'

            # Baca split type: 'percentage' atau 'crossval'
            split_type = split_cfg.get('type', 'percentage')
            test_ratio = float(split_cfg.get('test', 20)) / 100

            if 'crossval' in split_cfg and isinstance(split_cfg['crossval'], dict):
                n_folds = int(split_cfg['crossval'].get('folds', 10))
            else:
                n_folds = int(split_cfg.get('folds', 10))

            model_type = nb_params.get('modelType', 'MultinomialNB')
            feature_type = nb_params.get('feature', 'TfidfVectorizer')
            alpha = float(nb_params.get('alpha', 1.0))
            fit_prior = nb_params.get('fitPrior', 'True') == 'True'

            tiebreaker = params.get('tiebreaker', None)
            if tiebreaker is not None:
                # Tie‑breaker hanya untuk 7 emosi; non_idiom tidak diikutsertakan
                if set(tiebreaker.keys()) != ALLOWED_EMOTIONS or len(set(tiebreaker.values())) != 7:
                    raise ValueError("Ranking emosi tidak valid: harus mencakup 7 emosi (senang, sedih, marah, takut, terkejut, percaya, netral) dengan peringkat 1-7 yang unik.")
                log(f"Tie-breaker ranking diterima: {tiebreaker}", training_id)

            # ========== SIMPAN HOLD-OUT SET (DATA UJI) ==========
            original_texts = df['kalimat'].tolist()
            original_labels = df['label'].tolist()   # sekarang termasuk 'non_idiom'

            # Split data menjadi training (80%) dan hold-out (20%) dengan stratifikasi penuh
            train_texts, holdout_texts, train_labels, holdout_labels = train_test_split(
                original_texts, original_labels,
                test_size=test_ratio,
                random_state=random_state,
                stratify=original_labels
            )

            # Simpan hold-out set ke file CSV
            holdout_df = pd.DataFrame({
                'kalimat': holdout_texts,
                'label': holdout_labels
            })
            holdout_folder = 'data/testing'
            os.makedirs(holdout_folder, exist_ok=True)
            holdout_filename = f"holdout_lexicon_{training_id}.csv"
            holdout_path = os.path.join(holdout_folder, holdout_filename)
            holdout_df.to_csv(holdout_path, index=False)
            log(f"Hold-out set disimpan di {holdout_path}", training_id)

            # Mulai dari sini, GUNAKAN HANYA DATA TRAINING
            texts = [preprocess_text(t) for t in train_texts]
            labels = train_labels
            total_samples = len(texts)
            log(f"Jumlah sampel training: {total_samples}", training_id)
            log(f"Label unik: {set(labels)}", training_id)

            update_progress(app, training_id, 10, "Membuat vectorizer...")
            if feature_type == 'CountVectorizer':
                vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,2))
            else:
                vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

            log("Melakukan fit vectorizer...", training_id)
            X_text = vectorizer.fit_transform(texts)
            log(f"Shape fitur: {X_text.shape}", training_id)

            le = LabelEncoder()
            y = le.fit_transform(labels)
            classes = le.classes_.tolist()
            log(f"Label encoded: {classes}", training_id)

            update_progress(app, training_id, 25, "Membangun dictionary lexicon & PMI...")

            # 5. Bangun dictionary lexicon global (hanya dari data training)
            dict_lexicon = build_dictionary_lexicon(texts, labels, classes)
            log(f"Dictionary lexicon dibangun untuk {len(classes)} kelas", training_id)

            # 6. Frekuensi global untuk PMI (hanya dari data training)
            class_word_counts, class_doc_counts = compute_class_word_freq(texts, labels)
            total_docs = len(texts)
            vocab = set()
            for t in texts:
                vocab.update(t.split())
            vocab_size = len(vocab)

            # 7. Hitung lexicon scores (Dictionary + PMI) per sampel training
            lexicon_scores = []
            log("Menghitung skor lexicon (Dictionary + PMI)...", training_id)
            start = time.time()
            for i, text in enumerate(texts):
                words = text.split()
                dict_scores = compute_dictionary_score(words, dict_lexicon, classes)
                pmi_scores = compute_pmi_scores(words, classes, class_word_counts, class_doc_counts, total_docs, vocab_size)
                combined = dict_scores + pmi_scores
                lexicon_scores.append(combined)
                if (i+1) % 500 == 0 or i+1 == total_samples:
                    progress = 25 + int(((i+1)/total_samples)*15)
                    msg = f"Lexicon scoring: {i+1}/{total_samples}"
                    log(msg, training_id)
                    update_progress(app, training_id, progress, msg)

            lexicon_scores = np.array(lexicon_scores)
            log(f"Lexicon scores shape: {lexicon_scores.shape}", training_id)

            # ------------------------------------------------------------
            # PERCABANGAN: PERCENTAGE SPLIT vs CROSS-VALIDATION
            # ------------------------------------------------------------
            if split_type == 'crossval':
                # ===== TRUE K-FOLD CROSS VALIDATION =====
                log(f"Memulai {n_folds}-fold cross validation sejati", training_id)
                skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
                fold_metrics = []

                for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_text, y)):
                    X_tr = X_text[train_idx]
                    X_te = X_text[test_idx]
                    y_tr = y[train_idx]
                    y_te = y[test_idx]
                    lex_tr = lexicon_scores[train_idx]
                    lex_te = lexicon_scores[test_idx]

                    if model_type == 'BernoulliNB':
                        fold_model = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
                    else:
                        fold_model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                    fold_model.fit(X_tr, y_tr)
                    proba = fold_model.predict_proba(X_te)

                    # Fusion
                    method = fusion_params.get('method', 'product')
                    if method == 'sum':
                        final_scores = proba + lex_te
                    elif method == 'weighted':
                        w = float(fusion_params.get('weight', 0.5))
                        final_scores = w * proba + (1 - w) * lex_te
                    else:  # product
                        lex_te_norm = (lex_te - lex_te.min(axis=1, keepdims=True)) / (
                            lex_te.max(axis=1, keepdims=True) - lex_te.min(axis=1, keepdims=True) + 1e-8)
                        final_scores = proba * (1 + lex_te_norm)

                    exp_scores = np.exp(final_scores - np.max(final_scores, axis=1, keepdims=True))
                    final_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

                    if tiebreaker is not None:
                        y_pred = predict_with_tiebreaker(final_scores, classes, tiebreaker)
                    else:
                        y_pred = np.argmax(final_scores, axis=1)

                    acc = accuracy_score(y_te, y_pred)
                    prec_w = precision_score(y_te, y_pred, average='weighted', zero_division=0)
                    rec_w = recall_score(y_te, y_pred, average='weighted', zero_division=0)
                    f1_w = f1_score(y_te, y_pred, average='weighted')
                    fold_metrics.append({
                        'fold': fold_idx + 1,
                        'accuracy': round(acc, 4),
                        'precision': round(prec_w, 4),
                        'recall': round(rec_w, 4),
                        'f1_score': round(f1_w, 4),
                        'mcc': None
                    })
                    update_progress(app, training_id,
                                    45 + int((fold_idx+1)/n_folds * 30),
                                    f"Fold {fold_idx+1}/{n_folds} selesai (Akurasi: {acc:.4f})")

                if model_type == 'BernoulliNB':
                    final_model = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
                else:
                    final_model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                final_model.fit(X_text, y)
                model = final_model

                avg_acc = np.mean([f['accuracy'] for f in fold_metrics])
                avg_prec = np.mean([f['precision'] for f in fold_metrics])
                avg_rec = np.mean([f['recall'] for f in fold_metrics])
                avg_f1 = np.mean([f['f1_score'] for f in fold_metrics])

                metrics = {
                    'accuracy': round(avg_acc, 4),
                    'f1_score': round(avg_f1, 4),
                    'precision': round(avg_prec, 4),
                    'recall': round(avg_rec, 4),
                    'confusion_matrix': [],
                    'class_labels': classes,
                    'fold_metrics': fold_metrics,
                    'holdout_path': holdout_path
                }

            else:
                # ===== PERCENTAGE SPLIT (TETAP SEPERTI SEBELUMNYA) =====
                log(f"Menggunakan percentage split berbasis fold: test_ratio={test_ratio}, n_folds={n_folds}", training_id)
                n_test_folds = int(round(test_ratio * n_folds))
                if n_test_folds < 1:
                    n_test_folds = 1
                if n_test_folds >= n_folds:
                    n_test_folds = n_folds - 1

                log(f"StratifiedKFold (n_splits={n_folds}) dengan {n_test_folds} fold untuk testing", training_id)
                skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
                test_indices_per_fold = [test_idx for _, test_idx in skf.split(X_text, y)]
                test_idx = np.concatenate(test_indices_per_fold[:n_test_folds])
                train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

                X_train = X_text[train_idx]
                X_test = X_text[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                lex_train = lexicon_scores[train_idx]
                lex_test = lexicon_scores[test_idx]

                log(f"Data split: train={X_train.shape[0]}, test={X_test.shape[0]}", training_id)

                if model_type == 'BernoulliNB':
                    model = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
                else:
                    model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)
                log("Model NB selesai dilatih", training_id)

                update_progress(app, training_id, 65, "Melakukan fusi dengan lexicon...")

                method = fusion_params.get('method', 'product')
                if method == 'sum':
                    final_scores = proba + lex_test
                elif method == 'weighted':
                    w = float(fusion_params.get('weight', 0.5))
                    final_scores = w * proba + (1 - w) * lex_test
                else:  # product
                    lex_test_norm = (lex_test - lex_test.min(axis=1, keepdims=True)) / (
                        lex_test.max(axis=1, keepdims=True) - lex_test.min(axis=1, keepdims=True) + 1e-8)
                    final_scores = proba * (1 + lex_test_norm)

                exp_scores = np.exp(final_scores - np.max(final_scores, axis=1, keepdims=True))
                final_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

                if tiebreaker is not None:
                    log("Menggunakan tie-breaker ranking untuk prediksi.", training_id)
                    y_pred = predict_with_tiebreaker(final_scores, classes, tiebreaker)
                else:
                    y_pred = np.argmax(final_scores, axis=1)

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
                    'class_labels': classes,
                    'holdout_path': holdout_path
                }
                log(f"Evaluasi selesai. Akurasi: {metrics['accuracy']}, F1: {metrics['f1_score']}", training_id)

            # 13. Simpan model (berlaku untuk kedua mode)
            model_filename = f"lexicon_nb_{training_id}.pkl"
            model_path = os.path.join(MODEL_FOLDER, model_filename)
            artifacts = {
                'model': model,
                'vectorizer': vectorizer,
                'label_encoder': le,
                'dict_lexicon': dict_lexicon,           # ← tambahkan
                'class_word_counts': class_word_counts, # ← tambahkan
                'class_doc_counts': class_doc_counts,   # ← tambahkan
                'vocab_size': vocab_size,               # ← tambahkan
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