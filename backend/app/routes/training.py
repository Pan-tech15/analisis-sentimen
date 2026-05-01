import os
import uuid
import threading
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app import db
from app.models.model_config import ModelConfig
from app.models.training import Training
from app.services.indobert_knn import train_indobert_knn
from app.services.lexicon_nb import train_lexicon_nb
from app.models.preprocessing import Preprocessing

training_bp = Blueprint('training', __name__, url_prefix='/api/training')

UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@training_bp.route('/upload', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format file tidak didukung (hanya CSV/Excel)'}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    return jsonify({
        'filename': unique_name,
        'original_name': filename,
        'path': filepath
    }), 200

@training_bp.route('/start', methods=['POST'])
def start_training():
    data = request.get_json()
    config_id = data.get('config_id')
    dataset_path = data.get('dataset_path')

    if not config_id or not dataset_path:
        return jsonify({'error': 'config_id dan dataset_path diperlukan'}), 400

    config = ModelConfig.query.get(config_id)
    if not config:
        return jsonify({'error': 'Konfigurasi tidak ditemukan'}), 404
    
    dataset_id = None
    try:
        preprocessing = Preprocessing.query.filter_by(preprocessed_filepath=dataset_path).first()
        if preprocessing:
            dataset_id = preprocessing.dataset_id
    except:
        pass

    training = Training(
        model_config_id=config_id,
        dataset_filename=os.path.basename(dataset_path),
        dataset_id=dataset_id,
        status='pending'
    )
    db.session.add(training)
    db.session.commit()

    if config.algorithm == 'IndoBERT-KNN':
        train_func = train_indobert_knn
    elif config.algorithm == 'Lexicon-NB':
        train_func = train_lexicon_nb
    else:
        return jsonify({'error': f'Algoritma {config.algorithm} tidak didukung'}), 400

    app = current_app._get_current_object()
    thread = threading.Thread(target=train_func, args=(app, training.id, config, dataset_path))
    thread.start()

    return jsonify(training.to_dict()), 201

@training_bp.route('/status/<int:training_id>', methods=['GET'])
def get_status(training_id):
    training = Training.query.get(training_id)
    if not training:
        return jsonify({'error': 'Training tidak ditemukan'}), 404
    return jsonify(training.to_dict()), 200

@training_bp.route('/history', methods=['GET'])
def get_history():
    trainings = Training.query.order_by(Training.created_at.desc()).all()
    return jsonify([t.to_dict() for t in trainings]), 200

@training_bp.route('/<int:training_id>', methods=['DELETE'])
def delete_training(training_id):
    training = Training.query.get(training_id)
    if not training:
        return jsonify({'error': 'Training tidak ditemukan'}), 404
    db.session.delete(training)
    db.session.commit()
    return jsonify({'message': 'Training dihapus'}), 200

# HANYA SATU DEFINISI get_training_detail DENGAN LENGKAP
@training_bp.route('/<int:training_id>', methods=['GET'])
def get_training_detail(training_id):
    training = Training.query.get(training_id)
    if not training:
        return jsonify({'error': 'Training tidak ditemukan'}), 404
    result = training.to_dict()
    result['config_name'] = training.config.name if training.config else None
    result['algorithm'] = training.config.algorithm if training.config else None
    result['model_config_id'] = training.model_config_id
    result['params'] = training.config.params if training.config else None
    return jsonify(result), 200