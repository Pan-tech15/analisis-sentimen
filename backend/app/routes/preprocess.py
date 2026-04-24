import os
import pandas as pd
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_file
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
from app import db
from app.models.dataset import Dataset
from app.models.preprocessing import Preprocessing
from app.utils.preprocessing_utils import preprocess_text

preprocess_bp = Blueprint('preprocess', __name__, url_prefix='/api/preprocess')

def get_preprocessed_folder():
    backend_dir = os.path.dirname(current_app.root_path)
    folder = os.path.join(backend_dir, 'data', 'preprocessed')
    os.makedirs(folder, exist_ok=True)
    return folder

@preprocess_bp.route('/start', methods=['POST'])
def start_preprocessing():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Data JSON tidak valid'}), 400

    dataset_id = data.get('dataset_id')
    if not dataset_id:
        return jsonify({'error': 'dataset_id diperlukan'}), 400

    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'error': 'Dataset tidak ditemukan'}), 404

    raw_path = dataset.filepath
    if not os.path.exists(raw_path):
        return jsonify({'error': 'File dataset tidak ditemukan di server'}), 404

    try:
        df = pd.read_csv(raw_path)
        if 'kalimat' not in df.columns:
            return jsonify({'error': "Kolom 'kalimat' tidak ditemukan"}), 400

        df['cleaned_kalimat'] = df['kalimat'].apply(lambda x: preprocess_text(x))

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_name = os.path.splitext(dataset.filename)[0]
        preprocessed_filename = f"preprocessed_{timestamp}_{original_name}.csv"
        preprocessed_folder = get_preprocessed_folder()
        preprocessed_path = os.path.join(preprocessed_folder, preprocessed_filename)
        df.to_csv(preprocessed_path, index=False)

        # Hitung nomor urut preprocessing untuk dataset ini jako nama default
        existing_count = Preprocessing.query.filter_by(dataset_id=dataset_id).count()
        default_name = f"Preprocess #{existing_count + 1}"

        preprocessing = Preprocessing(
            dataset_id=dataset_id,
            preprocessed_filepath=preprocessed_path,
            row_count=len(df),
            name=default_name,       # NAMA DEFAULT
            status='completed',      # langsung completed karena proses sinkron
            progress=100
        )
        db.session.add(preprocessing)
        db.session.commit()

        return jsonify({
            'message': 'Preprocessing berhasil',
            'preprocessed_id': preprocessing.id,
            'filepath': preprocessed_path,
            'row_count': len(df),
            'name': preprocessing.name
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@preprocess_bp.route('/status/<int:preprocess_id>', methods=['GET'])
def get_preprocess_status(preprocess_id):
    preprocessing = Preprocessing.query.get(preprocess_id)
    if not preprocessing:
        return jsonify({'error': 'Preprocessing tidak ditemukan'}), 404

    return jsonify({
        'id': preprocessing.id,
        'status': preprocessing.status,
        'progress': preprocessing.progress,
        'row_count': preprocessing.row_count,
        'filepath': preprocessing.preprocessed_filepath
    }), 200


@preprocess_bp.route('/download/<int:preprocess_id>', methods=['GET', 'OPTIONS'])
@cross_origin(origins=["http://localhost:5500", "http://127.0.0.1:5500"], supports_credentials=True)
def download_preprocessed(preprocess_id):
    if request.method == 'OPTIONS':
        return '', 200

    preprocessing = Preprocessing.query.get(preprocess_id)
    if not preprocessing:
        return jsonify({'error': 'Hasil preprocessing tidak ditemukan'}), 404

    filepath = preprocessing.preprocessed_filepath

    # Pencarian file di beberapa lokasi alternatif
    if not os.path.exists(filepath):
        filename = os.path.basename(filepath)
        preprocessed_folder = get_preprocessed_folder()
        corrected_path = os.path.join(preprocessed_folder, filename)
        if os.path.exists(corrected_path):
            filepath = corrected_path
        else:
            alt_paths = [
                os.path.join(os.getcwd(), 'data', 'preprocessed', filename),
                os.path.join(os.path.dirname(current_app.root_path), 'data', 'preprocessed', filename)
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    filepath = alt
                    break
            else:
                return jsonify({'error': f'File tidak ditemukan di server. Nama file: {filename}'}), 404

    return send_file(
        filepath,
        as_attachment=True,
        download_name=os.path.basename(filepath),
        mimetype='text/csv'
    )


@preprocess_bp.route('/history', methods=['GET'])
def get_history():
    preprocessings = Preprocessing.query.order_by(Preprocessing.timestamp.desc()).all()
    results = []
    for p in preprocessings:
        results.append({
            'id': p.id,
            'dataset_id': p.dataset_id,
            'dataset_name': p.dataset.filename if p.dataset else None,
            'name': p.name or f"Preprocessing #{p.id}",  # selalu kirim name
            'preprocessed_filepath': p.preprocessed_filepath,
            'row_count': p.row_count,
            'created_at': p.timestamp.isoformat() if p.timestamp else None,  # alias untuk frontend
            'timestamp': p.timestamp.isoformat() if p.timestamp else None   # tetap ada untuk backward compatibility
        })
    return jsonify(results), 200


# ------------------- ENDPOINT BARU: RENAME & DELETE ------------------
@preprocess_bp.route('/history/<int:preprocess_id>', methods=['PUT'])
def rename_preprocessing(preprocess_id):
    """Mengganti nama preprocessing (frontend menggunakan PUT)"""
    preprocessing = Preprocessing.query.get(preprocess_id)
    if not preprocessing:
        return jsonify({'error': 'Preprocessing not found'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Data JSON tidak valid'}), 400

    new_name = data.get('name', '').strip()
    if not new_name:
        return jsonify({'error': 'Name cannot be empty'}), 400

    preprocessing.name = new_name
    db.session.commit()
    return jsonify({
        'message': 'Name updated',
        'name': new_name,
        'id': preprocessing.id
    }), 200


@preprocess_bp.route('/history/<int:preprocess_id>', methods=['DELETE'])
def delete_preprocessing(preprocess_id):
    """Menghapus preprocessing (frontend menggunakan DELETE)"""
    preprocessing = Preprocessing.query.get(preprocess_id)
    if not preprocessing:
        return jsonify({'error': 'Preprocessing not found'}), 404

    # Hapus file fisik jika ada
    filepath = preprocessing.preprocessed_filepath
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError as e:
            # Log error, tapi tetap hapus record database
            current_app.logger.warning(f"Could not delete file {filepath}: {e}")

    db.session.delete(preprocessing)
    db.session.commit()
    return jsonify({'message': 'Preprocessing deleted', 'id': preprocess_id}), 200