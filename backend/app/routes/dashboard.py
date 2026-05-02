import os
import pandas as pd
from flask import Blueprint, jsonify, current_app
from flask_jwt_extended import jwt_required
from app import db
from app.models.dataset import Dataset
from app.models.idiom import Idiom
from app.models.training import Training
from app.models.model_config import ModelConfig
from sqlalchemy import func

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/api/dashboard')

@dashboard_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_dashboard_stats():
    # Total kalimat (seluruh dataset)
    total_sentences = db.session.query(func.sum(Dataset.row_count)).scalar() or 0

    # Total idiom dalam kamus
    total_idioms = Idiom.query.count()

    # Total model (training)
    total_models = Training.query.count()

    # Idiom vs Non-Idiom
    total_has_idiom = db.session.query(func.sum(Dataset.has_idiom_count)).scalar() or 0
    total_no_idiom = db.session.query(func.sum(Dataset.no_idiom_count)).scalar() or 0

    # Distribusi emosi (baca semua file dataset)
    emotion_counts = {}
    datasets = Dataset.query.all()
    for ds in datasets:
        if os.path.exists(ds.filepath):
            try:
                df = pd.read_csv(ds.filepath)
                if 'emotion' in df.columns:
                    counts = df['emotion'].value_counts().to_dict()
                    for emo, cnt in counts.items():
                        emotion_counts[emo] = emotion_counts.get(emo, 0) + cnt
            except Exception as e:
                current_app.logger.error(f"Gagal baca {ds.filepath}: {e}")
    
    # Pastikan 8 kelas standar ada
    expected = ['Happy', 'Trust', 'Surprise', 'Neutral', 'Anger', 'Sadness', 'Fear', 'No Idiom']
    for emo in expected:
        emotion_counts.setdefault(emo, 0)

    # Model terbaik per algoritma
    best_models = {}
    algorithms = ['IndoBERT-KNN', 'Lexicon-NB']
    for algo in algorithms:
        best = db.session.query(Training).join(ModelConfig).filter(
            ModelConfig.algorithm == algo,
            Training.status == 'completed',
            Training.metrics.isnot(None)
        ).order_by(
            func.nullif(Training.metrics['accuracy'].astext, 'null').cast(db.Float).desc()
        ).first()
        
        if best and best.metrics and 'accuracy' in best.metrics:
            duration = None
            if best.completed_at and best.created_at:
                delta = best.completed_at - best.created_at
                minutes = delta.total_seconds() // 60
                seconds = delta.total_seconds() % 60
                duration = f"{int(minutes)} min {int(seconds)} sec"
            
            best_models[algo] = {
                'accuracy': best.metrics.get('accuracy'),
                'precision': best.metrics.get('precision'),
                'recall': best.metrics.get('recall'),
                'f1_score': best.metrics.get('f1_score'),
                'created_at': best.created_at.isoformat() if best.created_at else None,
                'duration': duration
            }
        else:
            best_models[algo] = None

    return jsonify({
        'total_sentences': total_sentences,
        'total_idioms': total_idioms,
        'total_models': total_models,
        'idiom_count': total_has_idiom,
        'non_idiom_count': total_no_idiom,
        'emotion_distribution': emotion_counts,
        'best_models': best_models
    }), 200