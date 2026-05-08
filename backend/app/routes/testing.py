import threading
import logging
import joblib  # <-- tambahkan ini
from flask import Blueprint, request, jsonify, current_app
from app import db
from app.models.training import Training
from app.models.testing import Testing
from app.services.testing_service import (
    run_testing,
    predict_single_text_with_idiom,
    predict_single_text_with_idiom_lexicon
)

testing_bp = Blueprint('testing', __name__)
logger = logging.getLogger(__name__)

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
    
    logger.info(f"Prediction request received - model_id: {model_id}, text: {text[:50]}...")
    
    if not model_id or not text:
        logger.warning("Missing model_id or text in request")
        return jsonify({'error': 'model_id dan text diperlukan'}), 400

    training = Training.query.get(model_id)
    if not training:
        logger.error(f"Model with id {model_id} not found")
        return jsonify({'error': 'Model tidak ditemukan'}), 404

    logger.info(f"Using model: {training.config.algorithm if training.config else 'unknown'}, "
                f"training_id: {training.id}, status: {training.status}")

    try:
        if training.config.algorithm == 'IndoBERT-KNN':
            logger.info("Calling predict_single_text_with_idiom for IndoBERT-KNN")
            result = predict_single_text_with_idiom(training, text)
            logger.info(f"Prediction result: {result}")
            return jsonify(result), 200
        elif training.config.algorithm == 'Lexicon-NB':
            logger.info("Calling predict_single_text_with_idiom_lexicon for Lexicon-NB")
            result = predict_single_text_with_idiom_lexicon(training, text)
            logger.info(f"Prediction result: {result}")
            return jsonify(result), 200
        else:
            logger.error(f"Unsupported algorithm: {training.config.algorithm}")
            return jsonify({'error': 'Algoritma tidak didukung'}), 400
    except Exception as e:
        logger.exception(f"Error during prediction: {str(e)}")
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

       # Ambil roc_auc dari testing.metrics jika ada
    roc_auc = None
    if testing.metrics and 'roc_auc' in testing.metrics:
        roc_auc = testing.metrics['roc_auc']

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
        'roc_auc': roc_auc,   # ← tambahkan ini
        'tested_at': testing.tested_at.isoformat() if testing.tested_at else None,
        'dataset_filename': training.dataset_filename,
        'algorithm': config.algorithm if config else None,
    }
    return jsonify(result), 200

@testing_bp.route('/best-model', methods=['GET'])
def get_best_model():
    """Mengembalikan model_id dengan akurasi tertinggi"""
    from app.models.training import Training
    from app.models.testing import Testing
    # Cari testing dengan accuracy tertinggi
    best_test = Testing.query.filter(Testing.status == 'completed', Testing.accuracy.isnot(None)) \
                            .order_by(Testing.accuracy.desc()).first()
    if best_test:
        return jsonify({
            'model_id': best_test.training_id,
            'accuracy': best_test.accuracy,
            'algorithm': best_test.training.config.algorithm if best_test.training and best_test.training.config else None
        }), 200
    # Jika tidak ada testing, cari training terbaru yang selesai
    best_train = Training.query.filter(Training.status == 'completed', Training.metrics.isnot(None)) \
                               .order_by(Training.completed_at.desc()).first()
    if best_train:
        accuracy = best_train.metrics.get('accuracy') if best_train.metrics else None
        return jsonify({
            'model_id': best_train.id,
            'accuracy': accuracy,
            'algorithm': best_train.config.algorithm if best_train.config else None
        }), 200
    return jsonify({'error': 'Belum ada model yang tersedia'}), 404