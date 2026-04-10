import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from flask import current_app
from app import db
from app.models.training import Training

# Library ML
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import umap

MODEL_FOLDER = 'data/models'
os.makedirs(MODEL_FOLDER, exist_ok=True)


def train_indobert_knn(training_id, config, dataset_path):
    """
    Pipeline pelatihan IndoBERT-KNN-UMAP.
    Parameter:
        training_id (int): ID record training di database.
        config (ModelConfig): Objek konfigurasi model.
        dataset_path (str): Path file CSV dataset.
    """
    with current_app.app_context():
        training = Training.query.get(training_id)
        if not training:
            return
        training.status = 'running'
        db.session.commit()

        try:
            # 1. Baca dataset
            df = pd.read_csv(dataset_path)
            if 'kalimat' not in df.columns or 'emotion' not in df.columns:
                raise ValueError("Dataset harus memiliki kolom 'kalimat' dan 'emotion'")

            texts = df['kalimat'].astype(str).tolist()
            labels = df['emotion'].tolist()

            # 2. Ambil parameter dari config.params
            params = config.params

            # Parameter umum
            random_state = int(params.get('general', {}).get('randomState', 42))
            shuffle = params.get('general', {}).get('shuffle', 'yes') == 'yes'
            stratified = params.get('general', {}).get('stratified', 'yes') == 'yes'

            # Split config
            split_config = params.get('split', {})
            split_type = split_config.get('type', 'crossval')
            cv_folds = int(split_config.get('crossval', {}).get('folds', 5))

            # Parameter IndoBERT
            bert_params = params.get('indobert', {})
            model_name = bert_params.get('modelName', 'indobenchmark/indobert-base-p2')
            max_seq_length = int(bert_params.get('maxSeqLength', 128))
            pooling = bert_params.get('pooling', 'CLS')
            batch_size = int(bert_params.get('batchSize', 16))

            # Parameter UMAP
            umap_params = params.get('umap', {})
            n_components = int(umap_params.get('nComponents', 25))
            n_neighbors = int(umap_params.get('nNeighbors', 30))
            min_dist = float(umap_params.get('minDist', 0.1))
            metric = umap_params.get('metric', 'cosine')
            umap_random_state = int(umap_params.get('randomState', 42))

            # Parameter KNN
            knn_params = params.get('knn', {})
            k = int(knn_params.get('k', 7))
            knn_metric = knn_params.get('metric', 'cosine')
            weights = knn_params.get('weights', 'distance')
            algorithm = knn_params.get('algorithm', 'auto')
            leaf_size = int(knn_params.get('leafSize', 30))
            p = int(knn_params.get('p', 2))

            # Update progress awal
            training.progress = 10
            db.session.commit()

            # 3. Ekstraksi fitur IndoBERT
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()

            embeddings = []
            total_samples = len(texts)
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

                progress = 20 + int((i / total_samples) * 30)
                training.progress = min(progress, 50)
                db.session.commit()

            embeddings = np.vstack(embeddings)
            training.progress = 55
            db.session.commit()

            # 4. Reduksi dimensi UMAP
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=umap_random_state
            )
            embeddings_reduced = reducer.fit_transform(embeddings)
            training.progress = 70
            db.session.commit()

            # 5. Klasifikasi KNN dengan validasi
            X = embeddings_reduced
            y = np.array(labels)

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric=knn_metric,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                n_jobs=-1
            )

            if split_type == 'percentage':
                test_size = float(split_config.get('test', 20)) / 100.0
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state,
                    shuffle=shuffle, stratify=y_encoded if stratified else None
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
                    'confusion_matrix': cm
                }
                knn.fit(X, y_encoded)
            else:
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
                    'cv_folds': cv_folds
                }
                knn.fit(X, y_encoded)

            training.progress = 90
            db.session.commit()

            # 6. Simpan model
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

            training.status = 'completed'
            training.progress = 100
            training.metrics = metrics
            training.model_path = model_path
            training.completed_at = datetime.utcnow()
            db.session.commit()

        except Exception as e:
            training.status = 'failed'
            training.metrics = {'error': str(e)}
            training.progress = 0
            db.session.commit()
            raise e