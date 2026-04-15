from flask import Blueprint, request, jsonify
from app import db
from app.models.model_config import ModelConfig
# from flask_jwt_extended import jwt_required  # Komentari dulu jika belum setup login

processing_bp = Blueprint('processing', __name__, url_prefix='/api/processing')

@processing_bp.route('/configs', methods=['GET'])
# @jwt_required()  # Nonaktifkan sementara
def get_configs():
    configs = ModelConfig.query.order_by(ModelConfig.created_at.desc()).all()
    return jsonify([c.to_dict() for c in configs]), 200

@processing_bp.route('/configs', methods=['POST'])
# @jwt_required()
def create_config():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Data JSON tidak valid'}), 400

    name = data.get('name')
    algorithm = data.get('algorithm')
    params = data.get('params')

    if not name or not algorithm or params is None:
        return jsonify({'error': 'Field name, algorithm, dan params wajib diisi'}), 400

    if ModelConfig.query.filter_by(name=name).first():
        return jsonify({'error': 'Nama model sudah digunakan'}), 409

    try:
        ModelConfig.validate_params(algorithm, params)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    new_config = ModelConfig(name=name, algorithm=algorithm, params=params)
    db.session.add(new_config)
    db.session.commit()

    return jsonify(new_config.to_dict()), 201

@processing_bp.route('/configs/<int:config_id>', methods=['PUT'])
# @jwt_required()
def update_config(config_id):
    config = ModelConfig.query.get(config_id)
    if not config:
        return jsonify({'error': 'Konfigurasi tidak ditemukan'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Data JSON tidak valid'}), 400

    name = data.get('name')
    algorithm = data.get('algorithm')
    params = data.get('params')

    if name:
        existing = ModelConfig.query.filter(ModelConfig.name == name, ModelConfig.id != config_id).first()
        if existing:
            return jsonify({'error': 'Nama model sudah digunakan'}), 409
        config.name = name

    if algorithm:
        config.algorithm = algorithm

    if params is not None:
        try:
            ModelConfig.validate_params(config.algorithm, params)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        config.params = params

    db.session.commit()
    return jsonify(config.to_dict()), 200

@processing_bp.route('/configs/<int:config_id>', methods=['GET'])
# @jwt_required()  # aktifkan jika perlu
def get_config(config_id):
    config = ModelConfig.query.get(config_id)
    if not config:
        return jsonify({'error': 'Konfigurasi tidak ditemukan'}), 404
    return jsonify(config.to_dict()), 200

@processing_bp.route('/configs/<int:config_id>', methods=['DELETE'])
# @jwt_required()
def delete_config(config_id):
    config = ModelConfig.query.get(config_id)
    if not config:
        return jsonify({'error': 'Konfigurasi tidak ditemukan'}), 404

    db.session.delete(config)
    db.session.commit()
    return jsonify({'message': 'Konfigurasi berhasil dihapus'}), 200

# -------------------------------------------------------------------
# CATATAN UNTUK KELOMPOK LEXICON+NAIVEBAYES:
# -------------------------------------------------------------------
# 1. Pastikan model `ModelConfig` sudah mencakup field `params` (JSON).
# 2. Di dalam `validate_params`, tambahkan validasi untuk struktur
#    parameter Lexicon+NaiveBayes sesuai kebutuhan kelompok Anda.
# 3. Endpoint di atas bersifat umum, tidak bergantung pada algoritma.
#    Semua parameter disimpan mentah di kolom `params`.
# 4. Saat akan melakukan pelatihan (di halaman Latih), backend akan
#    membaca `params` dan menjalankan pipeline yang sesuai.
# -------------------------------------------------------------------