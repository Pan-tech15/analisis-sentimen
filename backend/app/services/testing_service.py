import pandas as pd
import numpy as np
import re
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from app.models.training import Training
from app.models.testing import Testing
from app.models.idiom import Idiom
from app import db

# ------------------ PREPROCESSING SAMA SEPERTI TRAINING ------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------ LEXICON FUNCTIONS (COPY DARI TRAINING) ------------------
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

# ------------------ PREDICT UNTUK LEXICON-NB ------------------
def predict_lexicon(texts, artifacts):
    vectorizer = artifacts['vectorizer']
    model = artifacts['model']
    label_encoder = artifacts['label_encoder']
    classes = artifacts['classes']
    dict_lexicon = artifacts.get('dict_lexicon')
    class_word_counts = artifacts.get('class_word_counts')
    class_doc_counts = artifacts.get('class_doc_counts')
    vocab_size = artifacts.get('vocab_size')
    tiebreaker = artifacts.get('tiebreaker')
    fusion_method = artifacts.get('fusion_method', 'product')
    fusion_weight = artifacts.get('fusion_weight', 0.5)

    # Preprocessing teks
    cleaned_texts = [preprocess_text(t) for t in texts]

    # Transform ke vektor
    X_tfidf = vectorizer.transform(cleaned_texts)

    # Dapatkan proba dari Naive Bayes
    proba = model.predict_proba(X_tfidf)

    # Jika ada komponen lexicon, hitung skor lexicon (dictionary + PMI)
    if dict_lexicon:
        total_docs = sum(class_doc_counts.values()) if class_doc_counts else 1
        lexicon_scores = []
        for text in cleaned_texts:
            words = text.split()
            dict_scores = compute_dictionary_score(words, dict_lexicon, classes)
            pmi_scores = compute_pmi_scores(words, classes, class_word_counts, class_doc_counts, total_docs, vocab_size)
            combined = dict_scores + pmi_scores
            lexicon_scores.append(combined)
        lexicon_scores = np.array(lexicon_scores)
    else:
        # Fallback: lexicon scores = 0
        lexicon_scores = np.zeros_like(proba)

    # Fusion
    if fusion_method == 'sum':
        final_scores = proba + lexicon_scores
    elif fusion_method == 'weighted':
        final_scores = fusion_weight * proba + (1 - fusion_weight) * lexicon_scores
    else:  # product
        # Normalisasi lexicon_scores ke rentang positif
        lex_norm = (lexicon_scores - lexicon_scores.min(axis=1, keepdims=True)) / (
            lexicon_scores.max(axis=1, keepdims=True) - lexicon_scores.min(axis=1, keepdims=True) + 1e-8)
        final_scores = proba * (1 + lex_norm)

    # Softmax dan prediksi
    exp_scores = np.exp(final_scores - np.max(final_scores, axis=1, keepdims=True))
    final_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    if tiebreaker is not None:
        y_pred = predict_with_tiebreaker(final_scores, classes, tiebreaker)
    else:
        y_pred = np.argmax(final_scores, axis=1)

    # Kembalikan label asli
    return label_encoder.inverse_transform(y_pred)

def predict_lexicon_proba(texts, artifacts):
    vectorizer = artifacts['vectorizer']
    model = artifacts['model']
    label_encoder = artifacts['label_encoder']
    classes = artifacts['classes']
    dict_lexicon = artifacts.get('dict_lexicon')
    class_word_counts = artifacts.get('class_word_counts')
    class_doc_counts = artifacts.get('class_doc_counts')
    vocab_size = artifacts.get('vocab_size')
    tiebreaker = artifacts.get('tiebreaker')
    fusion_method = artifacts.get('fusion_method', 'product')
    fusion_weight = artifacts.get('fusion_weight', 0.5)

    cleaned_texts = [preprocess_text(t) for t in texts]
    X_tfidf = vectorizer.transform(cleaned_texts)
    proba_nb = model.predict_proba(X_tfidf)   # probabilitas dari Naive Bayes

    if dict_lexicon:
        total_docs = sum(class_doc_counts.values()) if class_doc_counts else 1
        lexicon_scores = []
        for text in cleaned_texts:
            words = text.split()
            dict_scores = compute_dictionary_score(words, dict_lexicon, classes)
            pmi_scores = compute_pmi_scores(words, classes, class_word_counts, class_doc_counts, total_docs, vocab_size)
            combined = dict_scores + pmi_scores
            lexicon_scores.append(combined)
        lexicon_scores = np.array(lexicon_scores)
    else:
        lexicon_scores = np.zeros_like(proba_nb)

    # Fusion (sama seperti di fungsi predict_lexicon)
    if fusion_method == 'sum':
        final_scores = proba_nb + lexicon_scores
    elif fusion_method == 'weighted':
        final_scores = fusion_weight * proba_nb + (1 - fusion_weight) * lexicon_scores
    else:  # product
        lex_norm = (lexicon_scores - lexicon_scores.min(axis=1, keepdims=True)) / (
            lexicon_scores.max(axis=1, keepdims=True) - lexicon_scores.min(axis=1, keepdims=True) + 1e-8)
        final_scores = proba_nb * (1 + lex_norm)

    # Softmax untuk mendapatkan probabilitas akhir
    exp_scores = np.exp(final_scores - np.max(final_scores, axis=1, keepdims=True))
    proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Prediksi
    if tiebreaker:
        y_pred = predict_with_tiebreaker(proba, classes, tiebreaker)
    else:
        y_pred = np.argmax(proba, axis=1)

    return label_encoder.inverse_transform(y_pred), proba

# ------------------ PREDICT UNTUK INDOBERT-KNN ------------------
def predict_indobert(texts, artifacts):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = artifacts['tokenizer']
    bert_model = artifacts['bert_model']
    umap_reducer = artifacts.get('umap_reducer')
    knn_classifier = artifacts['knn_classifier']
    label_encoder = artifacts['label_encoder']
    pooling = artifacts['pooling']
    max_seq_length = artifacts['max_seq_length']
    use_umap = artifacts.get('use_umap', False)

    bert_model.to(device)
    bert_model.eval()

    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text, truncation=True, padding='max_length',
                max_length=max_seq_length, return_tensors='pt'
            ).to(device)
            if hasattr(bert_model, 'bert'):
                # Model adalah IndoBERTClassifier (fine‑tuned)
                outputs = bert_model.bert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            else:
                # Model adalah AutoModel (tanpa fine‑tuning)
                outputs = bert_model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            if pooling == 'CLS':
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif pooling == 'MEAN':
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            elif pooling == 'MAX':
                emb = outputs.last_hidden_state.max(dim=1).values.cpu().numpy()
            else:
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(emb[0])
    embeddings = np.vstack(embeddings)

    if use_umap and umap_reducer:
        embeddings = umap_reducer.transform(embeddings)

    y_pred = knn_classifier.predict(embeddings)
    return label_encoder.inverse_transform(y_pred)

def predict_indobert_proba(texts, artifacts):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = artifacts['tokenizer']
    bert_model = artifacts['bert_model']
    umap_reducer = artifacts.get('umap_reducer')
    knn_classifier = artifacts['knn_classifier']
    label_encoder = artifacts['label_encoder']
    pooling = artifacts['pooling']
    max_seq_length = artifacts['max_seq_length']
    use_umap = artifacts.get('use_umap', False)
    hybrid_method = artifacts.get('hybrid_method', 'none')
    hybrid_alpha = artifacts.get('hybrid_alpha', 0.7)

    bert_model.to(device)
    bert_model.eval()

    # Ekstraksi embedding
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text, truncation=True, padding='max_length',
                max_length=max_seq_length, return_tensors='pt'
            ).to(device)
            if hasattr(bert_model, 'bert'):
                # Model adalah IndoBERTClassifier (fine‑tuned)
                outputs = bert_model.bert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            else:
                # Model adalah AutoModel (tanpa fine‑tuning)
                outputs = bert_model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            if pooling == 'CLS':
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif pooling == 'MEAN':
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            elif pooling == 'MAX':
                emb = outputs.last_hidden_state.max(dim=1).values.cpu().numpy()
            else:
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(emb[0])
    embeddings = np.vstack(embeddings)

    if use_umap and umap_reducer:
        embeddings = umap_reducer.transform(embeddings)

    # Probabilitas KNN
    proba_knn = knn_classifier.predict_proba(embeddings)

    # Confidence adjustment (sama seperti di training)
    if hybrid_method in ['confidence', 'weighted']:
        dist, _ = knn_classifier.kneighbors(embeddings)
        mean_dist = np.mean(dist, axis=1)
        confidence = 1.0 / (1.0 + mean_dist)
        confidence = confidence.reshape(-1, 1)

        if hybrid_method == 'confidence':
            final_scores = proba_knn * confidence
        elif hybrid_method == 'weighted':
            final_scores = hybrid_alpha * proba_knn + (1 - hybrid_alpha) * confidence
    else:
        final_scores = proba_knn

    # Prediksi dari final_scores
    y_pred = np.argmax(final_scores, axis=1)
    return label_encoder.inverse_transform(y_pred), final_scores

# ------------------ PREDICT UNTUK SATU TEKS (INPUT TEXT) ------------------
def predict_single_text(training, text):
    artifacts = joblib.load(training.model_path)
    if training.config.algorithm == 'Lexicon-NB':
        preds = predict_lexicon([text], artifacts)
    elif training.config.algorithm == 'IndoBERT-KNN':
        preds = predict_indobert([text], artifacts)
    else:
        raise ValueError("Algoritma tidak didukung")
    return preds[0]

# ------------------ MAIN FUNCTION UNTUK TESTING DATASET ------------------
def run_testing(app, test_id):
    with app.app_context():
        test = Testing.query.get(test_id)
        if not test:
            return
        try:
            test.status = 'running'
            test.progress = 10
            db.session.commit()

            training = Training.query.get(test.training_id)
            if not training:
                raise ValueError("Training tidak ditemukan")

            holdout_path = training.metrics['holdout_path']
            df = pd.read_csv(holdout_path)
            texts = df['kalimat'].tolist()
            y_true = df['label'].tolist() if 'label' in df.columns else df['emotion'].tolist()

            artifacts = joblib.load(training.model_path)

            # Pilih fungsi prediksi dengan probabilitas
            if training.config.algorithm == 'Lexicon-NB':
                y_pred, proba = predict_lexicon_proba(texts, artifacts)
            elif training.config.algorithm == 'IndoBERT-KNN':
                y_pred, proba = predict_indobert_proba(texts, artifacts)
            else:
                raise ValueError("Algoritma tidak didukung")

            # Hitung metrik weighted
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

            # Hitung metrik macro
            macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            # Untuk macro_accuracy gunakan balanced_accuracy_score (lebih cocok)
            from sklearn.metrics import balanced_accuracy_score
            macro_acc = balanced_accuracy_score(y_true, y_pred)

            # MCC
            mcc_val = matthews_corrcoef(y_true, y_pred)

            # ROC-AUC (menggunakan probabilitas yang sudah disiapkan)
            # Pastikan semua kelas y_true ada di proba, jika tidak, roc_auc bisa error
            try:
                roc_auc = roc_auc_score(y_true, proba, multi_class='ovr', average='weighted')
            except ValueError:
                roc_auc = None

            # Simpan semua metrik ke kolom individual (backward compatible) + metrics JSON
            test.accuracy = float(acc)
            test.f1_score = float(f1)
            test.precision = float(prec)
            test.recall = float(rec)
            test.confusion_matrix = confusion_matrix(y_true, y_pred).tolist()

            test.metrics = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'macro_accuracy': float(macro_acc),
                'macro_precision': float(macro_prec),
                'macro_recall': float(macro_rec),
                'macro_f1_score': float(macro_f1),
                'mcc': float(mcc_val),
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'class_labels': list(np.unique(np.concatenate([y_true, y_pred]))),  # atau dari label encoder
            }

            test.status = 'completed'
            test.progress = 100
            db.session.commit()

        except Exception as e:
            test.status = 'failed'
            test.progress = 0
            db.session.commit()
            raise e
        
# ------------------ DETEKSI IDIOM & PREDIKSI UNTUK INPUT TEXT INDOBERT-KNN------------------
def check_idiom_in_text(text):
    """Cari idiom dalam teks mentah (case‑insensitive) dan kembalikan (idiom_text, meaning) atau None."""
    # Ambil semua idiom dari database
    idioms = Idiom.query.all()
    text_lower = text.lower()
    for idiom in idioms:
        if idiom.idiom_text.lower() in text_lower:
            return idiom.idiom_text, idiom.idiom_meaning
    return None

def predict_single_text_with_idiom(training, text):
    """
    Untuk IndoBERT‑KNN: cek idiom → jika ada lanjut prediksi emosi; jika tidak, kembalikan pesan.
    Mengembalikan dictionary.
    """
    idiom_result = check_idiom_in_text(text)
    if not idiom_result:
        return {
            'has_idiom': False,
            'message': 'Tidak terdeteksi idiom'
        }
    idiom_text, idiom_meaning = idiom_result

    # Lakukan prediksi emosi dengan teks yang SUDAH DIBERSIHKAN
    from app.utils.preprocessing_utils import preprocess_text   # pakai fungsi yang sama dengan training
    cleaned = preprocess_text(text)
    # Panggil fungsi prediksi IndoBERT yang sudah ada
    artifacts = joblib.load(training.model_path)
    emotion = predict_indobert([cleaned], artifacts)[0]   # ambil prediksi pertama

    return {
        'has_idiom': True,
        'idiom_text': idiom_text,
        'idiom_meaning': idiom_meaning,
        'emotion': emotion
    }

# ------------------ DETEKSI IDIOM & PREDIKSI UNTUK INPUT TEXT LEXICON-NB ------------------
def predict_single_text_with_idiom_lexicon(training, text):
    """
    Untuk Lexicon‑NB: cek idiom → jika ada lanjut prediksi emosi; jika tidak, kembalikan pesan.
    Mengembalikan dictionary.
    """
    idiom_result = check_idiom_in_text(text)
    if not idiom_result:
        return {
            'has_idiom': False,
            'message': 'Tidak terdeteksi idiom'
        }
    idiom_text, idiom_meaning = idiom_result

    # Lakukan prediksi emosi dengan teks yang SUDAH DIBERSIHKAN
    cleaned = preprocess_text(text)   # pakai fungsi preprocess_text yang sudah ada di file ini
    # Panggil fungsi prediksi Lexicon yang sudah ada
    artifacts = joblib.load(training.model_path)
    emotion = predict_lexicon([cleaned], artifacts)[0]   # ambil prediksi pertama

    return {
        'has_idiom': True,
        'idiom_text': idiom_text,
        'idiom_meaning': idiom_meaning,
        'emotion': emotion
    }