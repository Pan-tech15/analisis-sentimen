import threading
import joblib  # <-- tambahkan ini
from flask import Blueprint, request, jsonify, current_app
from app import db
from app.models.training import Training
from app.models.testing import Testing
from app.services.testing_service import run_testing, predict_single_text

testing_bp = Blueprint('testing', __name__, url_prefix='/api/testing')

@testing_bp.route('/start', methods=['POST'])
def start_testing():
    data = request.get_json()
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({'error': 'model_id diperlukan'}), 400

    training = Training.query.get(model_id)
    if not training:
        return jsonify({'error': 'Model training tidak ditemukan'}), 404

    if not training.metrics or 'holdout_path' not in training.metrics:
        return jsonify({'error': 'Model tidak memiliki hold‑out set'}), 400

    # Buat record testing
    testing = Testing(
        training_id=model_id,
        test_dataset_id=None,  # opsional karena kita pakai hold‑out set
        status='pending'
    )
    db.session.add(testing)
    db.session.commit()

    app = current_app._get_current_object()
    thread = threading.Thread(target=run_testing, args=(app, testing.id))
    thread.start()

    return jsonify({'test_id': testing.id}), 201

@testing_bp.route('/predict', methods=['POST'])
def predict_text():
    data = request.get_json()
    model_id = data.get('model_id')
    text = data.get('text') or data.get('idiom_text')
    if not model_id or not text:
        return jsonify({'error': 'model_id dan text diperlukan'}), 400

    training = Training.query.get(model_id)
    if not training:
        return jsonify({'error': 'Model tidak ditemukan'}), 404

    try:
        if training.config.algorithm == 'IndoBERT-KNN':
            from app.services.testing_service import predict_single_text_with_idiom
            result = predict_single_text_with_idiom(training, text)
            return jsonify(result), 200
        elif training.config.algorithm == 'Lexicon-NB':
            from app.services.testing_service import predict_single_text_with_idiom_lexicon
            result = predict_single_text_with_idiom_lexicon(training, text)
            return jsonify(result), 200
        else:
            # fallback (tidak seharusnya terjadi)
            emotion = predict_single_text(training, text)
            return jsonify({'emotion': emotion}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@testing_bp.route('/status/<int:testing_id>', methods=['GET'])
def get_status(testing_id):
    testing = Testing.query.get(testing_id)
    if not testing:
        return jsonify({'error': 'Testing tidak ditemukan'}), 404
    return jsonify(testing.to_dict()), 200

@testing_bp.route('/history', methods=['GET'])
def get_history():
    tests = Testing.query.order_by(Testing.tested_at.desc()).all()
    return jsonify([t.to_dict() for t in tests]), 200

@testing_bp.route('/<int:testing_id>', methods=['GET'])
def get_detail(testing_id):
    testing = Testing.query.get(testing_id)
    if not testing:
        return jsonify({'error': 'Testing tidak ditemukan'}), 404

    training = Training.query.get(testing.training_id)
    if not training:
        return jsonify({'error': 'Training terkait tidak ditemukan'}), 404

    config = training.config

    # Ambil class_labels dari training.metrics atau dari model artifacts
    class_labels = []
    if training.metrics and 'class_labels' in training.metrics:
        class_labels = training.metrics['class_labels']
    elif training.model_path:
        try:
            artifacts = joblib.load(training.model_path)
            le = artifacts.get('label_encoder')
            if le:
                class_labels = le.classes_.tolist()
            else:
                class_labels = artifacts.get('classes', [])
        except Exception:
            class_labels = []

    result = {
        'id': testing.id,
        'training_id': testing.training_id,
        'status': testing.status,
        'accuracy': testing.accuracy,
        'f1_score': testing.f1_score,
        'precision': testing.precision,
        'recall': testing.recall,
        'confusion_matrix': testing.confusion_matrix,
        'class_labels': class_labels,
        'tested_at': testing.tested_at.isoformat() if testing.tested_at else None,
        'dataset_filename': training.dataset_filename,
        'algorithm': config.algorithm if config else None,
    }
    return jsonify(result), 200