import threading
import os
import logging
import joblib  # <-- tambahkan ini
from flask import Blueprint, request, jsonify, current_app
from app import db
from flask import send_file
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
    result = []
    for t in tests:
        d = t.to_dict()
        # Tambahkan config_name dari training
        if t.training and t.training.config:
            d['config_name'] = t.training.config.name
        else:
            d['config_name'] = None
        # Tambahkan dataset_name dari training (opsional, untuk konsistensi)
        if t.training and t.training.dataset:
            d['dataset_name'] = t.training.dataset.dataset_name
        else:
            d['dataset_name'] = None
        result.append(d)
    return jsonify(result), 200

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
        'dataset_name': training.dataset.dataset_name if training.dataset else None, 
        'dataset_filename': training.dataset_filename,
        'algorithm': config.algorithm if config else None,
        'metrics': testing.metrics,   
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

@testing_bp.route('/ensemble-predict', methods=['POST'])
def ensemble_predict():
    """Prediksi menggunakan ensemble (IndoBERT-KNN + Lexicon-NB) dengan bobot per kelas"""
    from app.services.ensemble import EnsembleService
    from app.models.training import Training
    from app.models.model_config import ModelConfig
    
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Text required'}), 400
    
    def get_best_model_by_algorithm(algorithm):
        """Mencari training dengan akurasi tertinggi untuk algoritma tertentu."""
        # Join Training dengan ModelConfig, filter berdasarkan algorithm
        trainings = Training.query.join(ModelConfig).filter(
            ModelConfig.algorithm == algorithm,
            Training.status == 'completed'
        ).all()
        best = None
        best_acc = -1.0
        for t in trainings:
            acc = t.metrics.get('accuracy', 0) if t.metrics else 0
            if acc > best_acc:
                best_acc = acc
                best = t
        return best
    
    model_a = get_best_model_by_algorithm('IndoBERT-KNN')
    model_b = get_best_model_by_algorithm('Lexicon-NB')
    
    if not model_a or not model_b:
        return jsonify({'error': 'Model belum tersedia. Silakan latih kedua model terlebih dahulu.'}), 404
    
    # Inisialisasi ensemble service
    ensemble = EnsembleService(model_a, model_b)
    
    # Prediksi
    emotion, confidence, scores = ensemble.predict(text)
    
    # Cek idiom (gunakan fungsi yang sudah ada di testing_service)
    from app.services.testing_service import check_idiom_in_text
    idiom_result = check_idiom_in_text(text)
    has_idiom = idiom_result is not None
    idiom_text = idiom_result[0] if has_idiom else None
    idiom_meaning = idiom_result[1] if has_idiom else None
    
    return jsonify({
        'has_idiom': has_idiom,
        'emotion': emotion,
        'confidence': confidence,
        'scores': scores,
        'idiom_text': idiom_text,
        'idiom_meaning': idiom_meaning
    })


@testing_bp.route('/<int:testing_id>/download', methods=['GET'])
def download_testing_model(testing_id):
    import os
    from flask import send_file, current_app

    testing = Testing.query.get(testing_id)
    if not testing or not testing.training or not testing.training.model_path:
        return jsonify({'error': 'Model not found'}), 404

    model_path = testing.training.model_path

    # Ubah path relatif menjadi absolut (sama seperti di training.py)
    if not os.path.isabs(model_path):
        backend_dir = os.path.dirname(current_app.root_path)  # naik satu level dari folder 'app'
        model_path = os.path.join(backend_dir, model_path)

    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found on server'}), 404

    return send_file(
        model_path,
        as_attachment=True,
        download_name=f"model_{testing.training_id}.pkl"
    )