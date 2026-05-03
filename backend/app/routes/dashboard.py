import os
import pandas as pd
from flask import Blueprint, jsonify, current_app, request
from flask_jwt_extended import jwt_required
from app import db
from app.models.dataset import Dataset
from app.models.idiom import Idiom
from app.models.training import Training
from app.models.model_config import ModelConfig
from app.models.testing import Testing
from sqlalchemy import func

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/api/dashboard')


@dashboard_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_dashboard_stats():
    total_datasets = Dataset.query.count()
    total_sentences = db.session.query(func.sum(Dataset.row_count)).scalar() or 0
    total_idioms = Idiom.query.count()
    total_models = Training.query.count()
    total_has_idiom = db.session.query(func.sum(Dataset.has_idiom_count)).scalar() or 0
    total_no_idiom = db.session.query(func.sum(Dataset.no_idiom_count)).scalar() or 0

    # Mapping emosi ke Inggris lowercase
    emotion_mapping = {
        'senang': 'happy', 'bahagia': 'happy',
        'marah': 'anger', 'jengkel': 'anger',
        'sedih': 'sadness',
        'takut': 'fear',
        'percaya': 'trust',
        'terkejut': 'surprise',
        'netral': 'neutral',
        'tidak ada idiom': 'no idiom',
        'no idiom': 'no idiom',
        'happy': 'happy', 'anger': 'anger', 'sadness': 'sadness',
        'fear': 'fear', 'trust': 'trust', 'surprise': 'surprise', 'neutral': 'neutral'
    }
    
    emotion_counts = {}
    datasets = Dataset.query.all()
    for ds in datasets:
        if os.path.exists(ds.filepath):
            try:
                df = pd.read_csv(ds.filepath)
                if 'emotion' in df.columns:
                    for emo in df['emotion']:
                        emo_str = str(emo).strip().lower()
                        normalized = emotion_mapping.get(emo_str, emo_str)
                        emotion_counts[normalized] = emotion_counts.get(normalized, 0) + 1
            except Exception as e:
                current_app.logger.error(f"Gagal baca {ds.filepath}: {e}")
    
    expected_emotions = ['happy', 'trust', 'surprise', 'neutral', 'anger', 'sadness', 'fear', 'no idiom']
    for emo in expected_emotions:
        emotion_counts.setdefault(emo, 0)

    # Best models
    best_models = {}
    algorithms = ['IndoBERT-KNN', 'Lexicon-NB']
    for algo in algorithms:
        trainings = db.session.query(Training).join(ModelConfig).filter(
            ModelConfig.algorithm == algo,
            Training.status == 'completed',
            Training.metrics.isnot(None)
        ).all()
        
        best = None
        max_acc = -1
        for t in trainings:
            acc = t.metrics.get('accuracy', 0) if t.metrics else 0
            if acc is not None and acc > max_acc:
                max_acc = acc
                best = t
        
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

    total_trained_indobert = db.session.query(Training).join(
        ModelConfig, Training.model_config_id == ModelConfig.id
    ).filter(
        ModelConfig.algorithm == 'IndoBERT-KNN',
        Training.status == 'completed'
    ).count()

    total_trained_lexicon = db.session.query(Training).join(
        ModelConfig, Training.model_config_id == ModelConfig.id
    ).filter(
        ModelConfig.algorithm == 'Lexicon-NB',
        Training.status == 'completed'
    ).count()

    return jsonify({
        'total_sentences': total_sentences,
        'total_datasets': total_datasets,
        'total_idioms': total_idioms,
        'total_models': total_models,
        'idiom_count': total_has_idiom,
        'non_idiom_count': total_no_idiom,
        'emotion_distribution': emotion_counts,
        'best_models': best_models,
        'total_trained_indobert': total_trained_indobert,
        'total_trained_lexicon': total_trained_lexicon
    }), 200


@dashboard_bp.route('/train-test-comparison', methods=['GET'])
@jwt_required()
def get_train_test_comparison():
    algorithm = request.args.get('algorithm', 'IndoBERT-KNN')
    
    trainings = db.session.query(Training).join(ModelConfig).filter(
        ModelConfig.algorithm == algorithm,
        Training.status == 'completed',
        Training.metrics.isnot(None)
    ).all()
    
    best_training = None
    max_acc = -1
    for t in trainings:
        acc = t.metrics.get('accuracy', 0) if t.metrics else 0
        if acc is not None and acc > max_acc:
            max_acc = acc
            best_training = t
    
    if not best_training:
        return jsonify({'error': f'Tidak ada model {algorithm} yang selesai'}), 404
    
    testing_record = Testing.query.filter_by(
        training_id=best_training.id,
        status='completed'
    ).order_by(Testing.tested_at.desc()).first()
    
    train_metrics = best_training.metrics or {}
    test_metrics = testing_record.to_dict() if testing_record else {}
    
    return jsonify({
        'algorithm': algorithm,
        'training': {
            'accuracy': train_metrics.get('accuracy'),
            'precision': train_metrics.get('precision'),
            'recall': train_metrics.get('recall'),
            'f1_score': train_metrics.get('f1_score')
        },
        'testing': {
            'accuracy': test_metrics.get('accuracy'),
            'precision': test_metrics.get('precision'),
            'recall': test_metrics.get('recall'),
            'f1_score': test_metrics.get('f1_score')
        },
        'training_id': best_training.id,
        'testing_id': testing_record.id if testing_record else None
    }), 200