import os
import sys
import time
import traceback
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ML Libraries
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import umap

# Import db dan Training dari app
from app import db
from app.models.training import Training

MODEL_FOLDER = 'data/models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# 7 emosi standar (lowercase)
ALLOWED_EMOTIONS = {'senang', 'sedih', 'marah', 'takut', 'terkejut', 'percaya', 'netral'}

def log(message, training_id=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f"[{timestamp}]"
    if training_id:
        prefix += f" [Training {training_id}]"
    print(f"{prefix} {message}", file=sys.stdout, flush=True)

def clean_and_map_emotion(raw_value):
    """Membersihkan dan memetakan label ke 7 emosi standar."""
    if pd.isna(raw_value):
        return None
    s = str(raw_value).strip().lower()
    if s in ALLOWED_EMOTIONS:
        return s
    # Pemetaan tambahan jika perlu (misal singkatan)
    # Tidak ada pemetaan lain, jika tidak cocok kembalikan None
    return None

def train_indobert_knn(app, training_id, config, dataset_path):
    """Pipeline pelatihan IndoBERT-KNN-UMAP."""
    with app.app_context():
        training = Training.query.get(training_id)
        if not training:
            log(f"Training {training_id} tidak ditemukan", training_id)
            return
        training.status = 'running'
        db.session.commit()
        log(f"Memulai training dengan konfigurasi: {config.name}", training_id)

        try:
            # 1. Baca dataset
            log(f"Membaca dataset dari {dataset_path}", training_id)
            df = pd.read_csv(dataset_path)
            log(f"Dataset loaded: {len(df)} baris, kolom: {list(df.columns)}", training_id)

            if 'kalimat' not in df.columns or 'emotion' not in df.columns:
                raise ValueError("Dataset harus memiliki kolom 'kalimat' dan 'emotion'")

            # Bersihkan label
            df['emotion_clean'] = df['emotion'].apply(clean_and_map_emotion)
            invalid_labels = df[df['emotion_clean'].isna()]
            if not invalid_labels.empty:
                invalid_values = invalid_labels['emotion'].unique()
                raise ValueError(f"Dataset mengandung label emosi tidak valid: {invalid_values}. Hanya 7 emosi yang diizinkan: {ALLOWED_EMOTIONS}")

            texts = df['kalimat'].astype(str).tolist()
            labels = df['emotion_clean'].tolist()
            total_samples = len(texts)
            log(f"Jumlah sampel setelah validasi: {total_samples}", training_id)
            log(f"Label unik: {set(labels)}", training_id)

            # 2. Parameter
            params = config.params
            random_state = int(params.get('general', {}).get('randomState', 42))
            shuffle = params.get('general', {}).get('shuffle', 'yes') == 'yes'
            stratified = params.get('general', {}).get('stratified', 'yes') == 'yes'
            split_config = params.get('split', {})
            split_type = split_config.get('type', 'crossval')
            cv_folds = int(split_config.get('crossval', {}).get('folds', 5))

            # Sesuaikan cv_folds
            if split_type == 'crossval' and cv_folds > total_samples:
                log(f"Menyesuaikan cv_folds dari {cv_folds} menjadi {total_samples}", training_id)
                cv_folds = total_samples

            bert_params = params.get('indobert', {})
            model_name = bert_params.get('modelName', 'indobenchmark/indobert-base-p2')
            max_seq_length = int(bert_params.get('maxSeqLength', 128))
            pooling = bert_params.get('pooling', 'CLS')
            batch_size = int(bert_params.get('batchSize', 16))

            umap_params = params.get('umap', {})
            n_components = int(umap_params.get('nComponents', 25))
            n_neighbors = int(umap_params.get('nNeighbors', 30))
            min_dist = float(umap_params.get('minDist', 0.1))
            metric = umap_params.get('metric', 'cosine')
            umap_random_state = int(umap_params.get('randomState', 42))

            if n_neighbors >= total_samples:
                new_n_neighbors = max(2, total_samples - 1)
                log(f"Menyesuaikan n_neighbors UMAP: {n_neighbors} -> {new_n_neighbors}", training_id)
                n_neighbors = new_n_neighbors

            knn_params = params.get('knn', {})
            k = int(knn_params.get('k', 7))
            knn_metric = knn_params.get('metric', 'cosine')
            weights = knn_params.get('weights', 'distance')
            algorithm = knn_params.get('algorithm', 'auto')
            leaf_size = int(knn_params.get('leafSize', 30))
            p = int(knn_params.get('p', 2))

            if k > total_samples:
                k = max(1, total_samples - 1)
                log(f"Menyesuaikan K: {knn_params.get('k')} -> {k}", training_id)

            # Progress awal
            training.progress = 5
            training.metrics = {'progress_message': 'Memulai ekstraksi fitur IndoBERT...'}
            db.session.commit()
            log("Memulai ekstraksi fitur IndoBERT...", training_id)

            # 3. Load model
            log(f"Loading IndoBERT model: {model_name}", training_id)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            log(f"Device: {device}", training_id)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()
            training.progress = 10
            training.metrics['progress_message'] = 'Model dimuat, ekstraksi fitur...'
            db.session.commit()
            log("Model IndoBERT dimuat", training_id)

            # 4. Ekstraksi fitur
            embeddings = []
            start_time = time.time()
            for i in range(0, total_samples, batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                    max_length=max_seq_length, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = model(**encoded)
                if pooling == 'CLS':
                    batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif pooling == 'MEAN':
                    batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                elif pooling == 'MAX':
                    batch_emb = outputs.last_hidden_state.max(dim=1).values.cpu().numpy()
                else:
                    batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_emb)

                progress = 10 + int(((i + len(batch_texts)) / total_samples) * 40)
                elapsed = time.time() - start_time
                if i % (batch_size * 5) == 0 or i + len(batch_texts) >= total_samples:
                    msg = f"Ekstraksi fitur: {min(i+len(batch_texts), total_samples)}/{total_samples} (progress {progress}%, elapsed {elapsed:.1f}s)"
                    log(msg, training_id)
                    training.progress = progress
                    training.metrics['progress_message'] = msg
                    db.session.commit()

            embeddings = np.vstack(embeddings)
            log(f"Ekstraksi fitur selesai. Shape: {embeddings.shape}", training_id)
            training.progress = 55
            training.metrics['progress_message'] = f"Ekstraksi fitur selesai. Shape: {embeddings.shape}"
            db.session.commit()

            # 5. UMAP
            original_dim = embeddings.shape[1]
            if n_components > original_dim:
                n_components = original_dim
            if n_components > total_samples:
                n_components = total_samples
            log(f"Memulai UMAP (n_neighbors={n_neighbors}, n_components={n_components})", training_id)
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=umap_random_state,
                verbose=True
            )
            embeddings_reduced = reducer.fit_transform(embeddings)
            log(f"UMAP selesai. Shape reduced: {embeddings_reduced.shape}", training_id)
            training.progress = 70
            training.metrics['progress_message'] = f"Reduksi dimensi selesai. Dimensi: {embeddings_reduced.shape[1]}"
            db.session.commit()

            # 6. KNN
            X = embeddings_reduced
            y = np.array(labels)
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            log(f"Label encoded: {le.classes_}", training_id)

            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric=knn_metric,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                n_jobs=-1
            )

            training.progress = 75
            training.metrics['progress_message'] = "Memulai pelatihan KNN dan evaluasi..."
            db.session.commit()
            log("Memulai KNN dan evaluasi", training_id)

            # Evaluasi adaptif
            if split_type == 'percentage':
                test_size = float(split_config.get('test', 20)) / 100.0
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state,
                    shuffle=shuffle, stratify=y_encoded if stratified and len(np.unique(y_encoded)) > 1 else None
                )
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
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
                    'class_labels': le.classes_.tolist()
                }
                knn.fit(X, y_encoded)
            else:
                unique, counts = np.unique(y_encoded, return_counts=True)
                min_class_count = counts.min()
                max_possible_folds = min_class_count

                if max_possible_folds < 2:
                    log(f"Peringatan: Ada kelas dengan hanya {min_class_count} sampel. Beralih ke Percentage Split (80/20).", training_id)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=0.2, random_state=random_state,
                        shuffle=True, stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
                    )
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)
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
                        'class_labels': le.classes_.tolist(),
                        'eval_method': 'percentage_split_fallback'
                    }
                    knn.fit(X, y_encoded)
                else:
                    if cv_folds > max_possible_folds:
                        log(f"Menyesuaikan cv_folds dari {cv_folds} menjadi {max_possible_folds}", training_id)
                        cv_folds = max_possible_folds

                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
                    cv_scores = cross_val_score(knn, X, y_encoded, cv=skf, scoring='accuracy')
                    acc = cv_scores.mean()
                    y_pred_cv = cross_val_predict(knn, X, y_encoded, cv=skf)
                    f1 = f1_score(y_encoded, y_pred_cv, average='weighted')
                    precision = precision_score(y_encoded, y_pred_cv, average='weighted', zero_division=0)
                    recall = recall_score(y_encoded, y_pred_cv, average='weighted', zero_division=0)
                    cm = confusion_matrix(y_encoded, y_pred_cv).tolist()
                    metrics = {
                        'accuracy': round(acc, 4),
                        'f1_score': round(f1, 4),
                        'precision': round(precision, 4),
                        'recall': round(recall, 4),
                        'confusion_matrix': cm,
                        'cv_folds': cv_folds,
                        'class_labels': le.classes_.tolist()
                    }
                    knn.fit(X, y_encoded)

            log(f"Evaluasi selesai. Akurasi: {metrics['accuracy']}, F1: {metrics['f1_score']}", training_id)
            training.progress = 90
            training.metrics = metrics
            training.metrics['progress_message'] = f"Evaluasi selesai. Akurasi: {metrics['accuracy']}"
            db.session.commit()

            # 7. Simpan model
            model_filename = f"indobert_knn_{training_id}.pkl"
            model_path = os.path.join(MODEL_FOLDER, model_filename)
            artifacts = {
                'tokenizer': tokenizer,
                'bert_model': model,
                'umap_reducer': reducer,
                'knn_classifier': knn,
                'label_encoder': le,
                'config': params,
                'pooling': pooling,
                'max_seq_length': max_seq_length,
                'device': str(device)
            }
            joblib.dump(artifacts, model_path)
            log(f"Model disimpan di {model_path}", training_id)

            training.status = 'completed'
            training.progress = 100
            training.model_path = model_path
            training.completed_at = datetime.utcnow()
            db.session.commit()
            log(f"Training {training_id} selesai dengan sukses!", training_id)

        except Exception as e:
            error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
            log(error_msg, training_id)
            training.status = 'failed'
            training.metrics = {'error': str(e), 'progress_message': f'Gagal: {str(e)}'}
            training.progress = 0
            db.session.commit()
            raise e