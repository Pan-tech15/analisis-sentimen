# =========================
# FINAL VERSION - LEXICON + NB
# =========================

import os
import sys
import re
import traceback
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from flask import current_app
from app import db
from app.models.training import Training

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

MODEL_FOLDER = 'data/models'
os.makedirs(MODEL_FOLDER, exist_ok=True)


# =========================
# PREPROCESSING
# =========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =========================
# LOAD LEXICON
# =========================
def load_lexicon(filepath):
    df = pd.read_csv(filepath)
    lexicon = {}
    for _, row in df.iterrows():
        word = str(row['kata']).lower().strip()
        emotion = str(row['emosi']).lower().strip()
        score = float(row['skor'])
        if word not in lexicon:
            lexicon[word] = {}
        lexicon[word][emotion] = score
    return lexicon


# =========================
# PMI
# =========================
def compute_class_word_freq(texts, labels):
    class_word_counts = {}
    class_doc_counts = {}

    for text, label in zip(texts, labels):
        words = text.split()  # ✅ FIX (bukan set)
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


# =========================
# TRAINING
# =========================
def train_lexicon_nb(training_id, config, dataset_path):
    with current_app.app_context():
        training = Training.query.get(training_id)

        try:
            df = pd.read_csv(dataset_path)

            texts = [preprocess_text(t) for t in df['kalimat']]
            labels = df['emotion'].tolist()

            params = config.params

            # =========================
            # PARAMETER
            # =========================
            general = params.get('general', {})
            split = params.get('split', {})
            nb_params = params.get('naivebayes', {})
            fusion_params = params.get('fusion', {})

            random_state = int(general.get('randomState', 42))
            shuffle = general.get('shuffle', 'yes') == 'yes'
            stratified = general.get('stratified', 'yes') == 'yes'

            split_type = split.get('type', 'percentage')
            test_ratio = float(split.get('test', 20)) / 100

            # =========================
            # NB PARAMS
            # =========================
            model_type = nb_params.get('modelType', 'MultinomialNB')
            feature_type = nb_params.get('feature', 'TfidfVectorizer')
            alpha = float(nb_params.get('alpha', 1.0))
            fit_prior = nb_params.get('fitPrior', 'True') == 'True'

            # =========================
            # VECTORIZER
            # =========================
            if feature_type == 'CountVectorizer':
                vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,2))
            else:
                vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

            X_text = vectorizer.fit_transform(texts)

            # =========================
            # LABEL
            # =========================
            le = LabelEncoder()
            y = le.fit_transform(labels)
            classes = le.classes_.tolist()

            # =========================
            # LEXICON
            # =========================
            lexicon = load_lexicon("data/lexicon.csv")  # ganti path

            total_docs = len(texts)
            class_word_counts, class_doc_counts = compute_class_word_freq(texts, labels)
            vocab_size = len(set(" ".join(texts).split()))

            # =========================
            # HITUNG LEXICON SCORE
            # =========================
            lexicon_scores = []

            for text in texts:
                words = text.split()
                vec = np.zeros(len(classes))

                for w in words:
                    for i, c in enumerate(classes):
                        lex = lexicon.get(w, {}).get(c, 0)
                        pmi = compute_pmi(w, c, class_word_counts, class_doc_counts, total_docs, vocab_size)
                        vec[i] += (lex + pmi)

                lexicon_scores.append(vec)

            lexicon_scores = np.array(lexicon_scores)

            # =========================
            # MODEL
            # =========================
            if model_type == 'BernoulliNB':
                model = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
            else:
                model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

            # =========================
            # SPLIT
            # =========================
            X_train, X_test, y_train, y_test, lex_train, lex_test = train_test_split(
                X_text, y, lexicon_scores,
                test_size=test_ratio,
                random_state=random_state,
                shuffle=shuffle,
                stratify=y if stratified else None
            )

            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)

            # =========================
            # FUSION
            # =========================
            method = fusion_params.get('method', 'product')

            if method == 'sum':
                final_scores = proba + lex_test

            elif method == 'weighted':
                w = float(fusion_params.get('weight', 0.5))
                final_scores = w * proba + (1 - w) * lex_test

            else:  # product
                final_scores = proba * (1 + lex_test)

            # =========================
            # NORMALISASI (INTENSITAS)
            # =========================
            exp_scores = np.exp(final_scores)
            final_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            y_pred = np.argmax(final_scores, axis=1)

            # =========================
            # METRICS
            # =========================
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average='weighted'),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classes": classes
            }

            # =========================
            # SAVE MODEL
            # =========================
            path = os.path.join(MODEL_FOLDER, f"model_{training_id}.pkl")

            joblib.dump({
                "model": model,
                "vectorizer": vectorizer,
                "label_encoder": le,
                "lexicon": lexicon,
                "classes": classes,
                "config": params
            }, path)

            training.status = "completed"
            training.model_path = path
            training.metrics = metrics
            db.session.commit()

        except Exception as e:
            training.status = "failed"
            training.metrics = {"error": str(e)}
            db.session.commit()
            raise e