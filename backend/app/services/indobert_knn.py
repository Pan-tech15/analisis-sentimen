import os
import sys
import time
import traceback
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ML Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import umap

from app import db
from app.models.training import Training

MODEL_FOLDER = 'data/models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# 7 emosi yang diizinkan (lowercase)
ALLOWED_EMOTIONS = {'senang', 'sedih', 'marah', 'takut', 'terkejut', 'percaya', 'netral'}

def log(message, training_id=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f"[{timestamp}]"
    if training_id:
        prefix += f" [Training {training_id}]"
    print(f"{prefix} {message}", file=sys.stdout, flush=True)

def clean_and_map_emotion(raw_value):
    """Bersihkan dan petakan label ke 7 emosi standar. None jika tidak valid."""
    if pd.isna(raw_value):
        return None
    s = str(raw_value).strip().lower()
    return s if s in ALLOWED_EMOTIONS else None

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class IndoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, freeze_layers=0, pooling='MEAN'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        # Bekukan layer tertentu
        if freeze_layers > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == 'CLS':
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'MEAN':
            pooled = outputs.last_hidden_state.mean(dim=1)
        elif self.pooling == 'MAX':
            pooled = outputs.last_hidden_state.max(dim=1).values
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

def train_indobert_knn(app, training_id, config, dataset_path):
    with app.app_context():
        training = Training.query.get(training_id)
        if not training:
            log(f"Training {training_id} tidak ditemukan", training_id)
            return
        training.status = 'running'
        db.session.commit()
        log(f"Memulai training dengan konfigurasi: {config.name}", training_id)

        try:
            # ========== 1. Baca dan bersihkan dataset ==========
            log(f"Membaca dataset dari {dataset_path}", training_id)
            df = pd.read_csv(dataset_path)
            log(f"Dataset loaded: {len(df)} baris, kolom: {list(df.columns)}", training_id)

            if 'kalimat' not in df.columns or 'emotion' not in df.columns:
                raise ValueError("Dataset harus memiliki kolom 'kalimat' dan 'emotion'")

            df['emotion_clean'] = df['emotion'].apply(clean_and_map_emotion)
            before = len(df)
            df = df.dropna(subset=['emotion_clean']).reset_index(drop=True)
            after = len(df)
            if before != after:
                log(f"Dibuang {before - after} baris karena label emosi tidak valid / nan.", training_id)

            if len(df) == 0:
                raise ValueError("Tidak ada data valid setelah membersihkan label emosi.")

            texts = df['kalimat'].astype(str).tolist()
            labels = df['emotion_clean'].tolist()
            total_samples = len(texts)
            log(f"Jumlah sampel setelah validasi: {total_samples}", training_id)
            log(f"Label unik: {set(labels)}", training_id)

            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(labels)
            num_classes = len(le.classes_)
            log(f"Label encoded: {le.classes_}", training_id)

            # ========== 2. Baca parameter dari config ==========
            params = config.params

            # --- Parameter umum ---
            general = params.get('general', {})
            random_state = int(general.get('randomState', 42))
            def to_bool(val, default=True):
                if isinstance(val, bool): return val
                if isinstance(val, str): return val.lower() in ('yes', 'true', '1')
                return default
            shuffle = to_bool(general.get('shuffle', 'yes'))
            stratified = to_bool(general.get('stratified', 'yes'))

            split_config = params.get('split', {})
            split_type = split_config.get('type', 'crossval')
            cv_folds = int(split_config.get('crossval', {}).get('folds', 5))
            if split_type == 'crossval' and cv_folds > total_samples:
                cv_folds = total_samples

            # --- Parameter IndoBERT ---
            bert_params = params.get('indobert', {})
            model_name = bert_params.get('modelName', 'indobenchmark/indobert-base-p2')
            max_seq_length = int(bert_params.get('maxSeqLength', 128))
            pooling = bert_params.get('pooling', 'MEAN')
            batch_size = int(bert_params.get('batchSize', 32))

            # --- Parameter Fine‑Tuning ---
            finetune_params = params.get('finetune', {})
            do_finetune = to_bool(finetune_params.get('enabled', False))
            ft_epochs = int(finetune_params.get('epochs', 3))
            ft_lr = float(finetune_params.get('learningRate', 2e-5))
            ft_optimizer = finetune_params.get('optimizer', 'AdamW')
            ft_weight_decay = float(finetune_params.get('weightDecay', 0.01))
            ft_warmup_ratio = float(finetune_params.get('warmupRatio', 0.1))
            ft_freeze_layers = int(finetune_params.get('freezeLayers', 9))
            ft_grad_accum = int(finetune_params.get('gradientAccumulation', 1))
            ft_max_grad_norm = float(finetune_params.get('maxGradNorm', 1.0))

            # --- Parameter UMAP ---
            umap_params = params.get('umap', {})
            use_umap = False
            if umap_params:
                umap_enabled = umap_params.get('enabled', False)
                if isinstance(umap_enabled, str):
                    use_umap = umap_enabled.lower() == 'true'
                else:
                    use_umap = bool(umap_enabled)
            n_components = int(umap_params.get('nComponents', 50))
            n_neighbors = int(umap_params.get('nNeighbors', 30))
            min_dist = float(umap_params.get('minDist', 0.1))
            umap_metric = umap_params.get('metric', 'cosine')
            umap_random_state = int(umap_params.get('randomState', 42))

            # --- Parameter KNN ---
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

            # --- Parameter Confidence Adjustment (Hybrid) ---
            hybrid_params = params.get('hybrid', {})
            hybrid_method = hybrid_params.get('method', 'none')
            hybrid_alpha = float(hybrid_params.get('alpha', 0.7))

            # ========== 3. Update progress awal ==========
            training.progress = 5
            training.metrics = {'progress_message': 'Memulai proses...'}
            db.session.commit()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            log(f"Device: {device}", training_id)

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # ========== 4. Fine‑Tuning (jika diaktifkan) ==========
            if do_finetune:
                log("Memulai Fine‑Tuning IndoBERT...", training_id)
                training.progress = 10
                training.metrics['progress_message'] = 'Fine‑Tuning IndoBERT...'
                db.session.commit()

                # Bagi data train/val untuk fine‑tuning (80/20)
                X_train, X_val, y_train, y_val = train_test_split(
                    texts, y_encoded, test_size=0.2, random_state=random_state,
                    stratify=y_encoded if stratified else None
                )
                train_dataset = EmotionDataset(X_train, y_train, tokenizer, max_seq_length)
                val_dataset = EmotionDataset(X_val, y_val, tokenizer, max_seq_length)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                model = IndoBERTClassifier(model_name, num_classes, freeze_layers=ft_freeze_layers, pooling=pooling)
                model.to(device)

                # Optimizer
                if ft_optimizer.lower() == 'adamw':
                    optimizer = AdamW(model.parameters(), lr=ft_lr, weight_decay=ft_weight_decay)
                elif ft_optimizer.lower() == 'adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr, weight_decay=ft_weight_decay)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=ft_lr, weight_decay=ft_weight_decay)

                total_steps = len(train_loader) * ft_epochs // ft_grad_accum
                warmup_steps = int(total_steps * ft_warmup_ratio)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

                criterion = nn.CrossEntropyLoss()

                for epoch in range(ft_epochs):
                    model.train()
                    total_loss = 0
                    for step, batch in enumerate(train_loader):
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        logits = model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                        loss = loss / ft_grad_accum
                        loss.backward()
                        if (step + 1) % ft_grad_accum == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), ft_max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                        total_loss += loss.item() * ft_grad_accum

                    avg_loss = total_loss / len(train_loader)
                    log(f"Epoch {epoch+1}/{ft_epochs} - Loss: {avg_loss:.4f}", training_id)
                    training.progress = 10 + int((epoch+1) / ft_epochs * 20)
                    training.metrics['progress_message'] = f'Fine‑Tuning epoch {epoch+1}/{ft_epochs}'
                    db.session.commit()

                # Simpan model fine‑tuned untuk ekstraksi fitur nanti
                ft_model = model
                log("Fine‑Tuning selesai.", training_id)
            else:
                # Load model pre‑trained tanpa fine‑tuning
                ft_model = AutoModel.from_pretrained(model_name).to(device)
                log("Menggunakan model pre‑trained tanpa fine‑tuning.", training_id)

            # ========== 5. Ekstraksi fitur (embedding) ==========
            training.progress = 30
            training.metrics['progress_message'] = 'Ekstraksi fitur...'
            db.session.commit()
            log("Memulai ekstraksi fitur...", training_id)

            ft_model.eval()
            embeddings = []
            start_time = time.time()
            with torch.no_grad():
                for i in range(0, total_samples, batch_size):
                    batch_texts = texts[i:i+batch_size]
                    encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                        max_length=max_seq_length, return_tensors='pt').to(device)
                    if do_finetune:
                        # Gunakan output dari model classifier (sebelum linear) atau dari BERT
                        # Kita bisa mengambil representasi dari layer sebelum classifier
                        outputs = ft_model.bert(**encoded)
                    else:
                        outputs = ft_model(**encoded)
                    if pooling == 'CLS':
                        batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    elif pooling == 'MEAN':
                        batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    elif pooling == 'MAX':
                        batch_emb = outputs.last_hidden_state.max(dim=1).values.cpu().numpy()
                    else:
                        batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(batch_emb)

                    progress = 30 + int(((i + len(batch_texts)) / total_samples) * 30)
                    if i % (batch_size * 5) == 0 or i + len(batch_texts) >= total_samples:
                        elapsed = time.time() - start_time
                        msg = f"Ekstraksi: {min(i+len(batch_texts), total_samples)}/{total_samples} (progress {progress}%)"
                        log(msg, training_id)
                        training.progress = progress
                        training.metrics['progress_message'] = msg
                        db.session.commit()

            embeddings = np.vstack(embeddings)
            log(f"Ekstraksi fitur selesai. Shape: {embeddings.shape}", training_id)

            # ========== 6. UMAP (opsional) ==========
            reducer = None
            if use_umap:
                original_dim = embeddings.shape[1]
                if n_components > original_dim:
                    n_components = original_dim
                if n_components > total_samples:
                    n_components = total_samples
                if n_neighbors >= total_samples:
                    n_neighbors = max(2, total_samples - 1)
                log(f"UMAP: n_neighbors={n_neighbors}, n_components={n_components}", training_id)
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=umap_metric,
                    random_state=umap_random_state,
                    verbose=True
                )
                X = reducer.fit_transform(embeddings)
                log(f"UMAP selesai. Shape: {X.shape}", training_id)
            else:
                X = embeddings
                log("UMAP dilewati.", training_id)

            training.progress = 70
            training.metrics['progress_message'] = 'Memulai KNN dan evaluasi...'
            db.session.commit()

            # ========== 7. KNN dengan Confidence Adjustment ==========
            knn = KNeighborsClassifier(
                n_neighbors=k, metric=knn_metric, weights=weights,
                algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=-1
            )

            # Evaluasi sesuai split_type (sama seperti sebelumnya, tidak diubah)
            if split_type == 'percentage':
                test_size = float(split_config.get('test', 20)) / 100.0
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state,
                    shuffle=shuffle, stratify=y_encoded if stratified and len(np.unique(y_encoded))>1 else None
                )
                knn.fit(X_train, y_train)
                proba_knn = knn.predict_proba(X_test)
                dist, _ = knn.kneighbors(X_test)
                mean_dist = np.mean(dist, axis=1)
                confidence = 1.0 / (1.0 + mean_dist)
                confidence = confidence.reshape(-1, 1)

                if hybrid_method == 'confidence':
                    final_scores = proba_knn * confidence
                elif hybrid_method == 'weighted':
                    final_scores = hybrid_alpha * proba_knn + (1 - hybrid_alpha) * confidence
                else:
                    final_scores = proba_knn

                y_pred = np.argmax(final_scores, axis=1)
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
                    'hybrid_method': hybrid_method,
                    'use_umap': use_umap,
                    'finetuned': do_finetune
                }
                knn.fit(X, y_encoded)
            else:
                # Cross-validation (kode sama seperti sebelumnya, dengan penambahan metrics)
                unique, counts = np.unique(y_encoded, return_counts=True)
                min_class_count = counts.min()
                max_possible_folds = min_class_count
                if max_possible_folds < 2:
                    log("Fallback ke percentage split karena kelas tidak cukup.", training_id)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=0.2, random_state=random_state, shuffle=True,
                        stratify=y_encoded if len(np.unique(y_encoded))>1 else None
                    )
                    knn.fit(X_train, y_train)
                    proba_knn = knn.predict_proba(X_test)
                    dist, _ = knn.kneighbors(X_test)
                    mean_dist = np.mean(dist, axis=1)
                    confidence = 1.0 / (1.0 + mean_dist).reshape(-1, 1)
                    if hybrid_method == 'confidence':
                        final_scores = proba_knn * confidence
                    elif hybrid_method == 'weighted':
                        final_scores = hybrid_alpha * proba_knn + (1 - hybrid_alpha) * confidence
                    else:
                        final_scores = proba_knn
                    y_pred = np.argmax(final_scores, axis=1)
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
                        'hybrid_method': hybrid_method,
                        'use_umap': use_umap,
                        'finetuned': do_finetune,
                        'eval_method': 'percentage_split_fallback'
                    }
                    knn.fit(X, y_encoded)
                else:
                    if cv_folds > max_possible_folds:
                        cv_folds = max_possible_folds
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
                    proba_list, conf_list, y_true_list = [], [], []
                    for train_idx, test_idx in skf.split(X, y_encoded):
                        X_tr, X_te = X[train_idx], X[test_idx]
                        y_tr, y_te = y_encoded[train_idx], y_encoded[test_idx]
                        knn_cv = KNeighborsClassifier(
                            n_neighbors=k, metric=knn_metric, weights=weights,
                            algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=-1
                        )
                        knn_cv.fit(X_tr, y_tr)
                        proba = knn_cv.predict_proba(X_te)
                        dist, _ = knn_cv.kneighbors(X_te)
                        mean_dist = np.mean(dist, axis=1)
                        conf = 1.0 / (1.0 + mean_dist)
                        proba_list.append(proba)
                        conf_list.append(conf.reshape(-1, 1))
                        y_true_list.append(y_te)
                    proba_knn = np.vstack(proba_list)
                    confidence = np.vstack(conf_list)
                    y_true_all = np.concatenate(y_true_list)
                    if hybrid_method == 'confidence':
                        final_scores = proba_knn * confidence
                    elif hybrid_method == 'weighted':
                        final_scores = hybrid_alpha * proba_knn + (1 - hybrid_alpha) * confidence
                    else:
                        final_scores = proba_knn
                    y_pred = np.argmax(final_scores, axis=1)
                    acc = accuracy_score(y_true_all, y_pred)
                    f1 = f1_score(y_true_all, y_pred, average='weighted')
                    precision = precision_score(y_true_all, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_true_all, y_pred, average='weighted', zero_division=0)
                    cm = confusion_matrix(y_true_all, y_pred).tolist()
                    metrics = {
                        'accuracy': round(acc, 4),
                        'f1_score': round(f1, 4),
                        'precision': round(precision, 4),
                        'recall': round(recall, 4),
                        'confusion_matrix': cm,
                        'cv_folds': cv_folds,
                        'class_labels': le.classes_.tolist(),
                        'hybrid_method': hybrid_method,
                        'use_umap': use_umap,
                        'finetuned': do_finetune
                    }
                    knn.fit(X, y_encoded)

            log(f"Evaluasi selesai. Akurasi: {metrics['accuracy']}, F1: {metrics['f1_score']}", training_id)

            # ========== 8. Simpan model ==========
            model_filename = f"indobert_knn_{training_id}.pkl"
            model_path = os.path.join(MODEL_FOLDER, model_filename)
            artifacts = {
                'tokenizer': tokenizer,
                'bert_model': ft_model,  # bisa jadi classifier atau bare model
                'umap_reducer': reducer,
                'knn_classifier': knn,
                'label_encoder': le,
                'config': params,
                'pooling': pooling,
                'max_seq_length': max_seq_length,
                'device': str(device),
                'use_umap': use_umap,
                'hybrid_method': hybrid_method,
                'finetuned': do_finetune
            }
            joblib.dump(artifacts, model_path)
            log(f"Model disimpan di {model_path}", training_id)

            training.status = 'completed'
            training.progress = 100
            training.model_path = model_path
            training.metrics = metrics
            training.metrics['progress_message'] = f"Selesai. Akurasi: {metrics['accuracy']}"
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