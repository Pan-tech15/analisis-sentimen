import os
import pandas as pd
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_file
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import threading
from datetime import datetime
from app import db
from app.models.dataset import Dataset
from app.models.preprocessing import Preprocessing
from app.utils.preprocessing_utils import preprocess_light

preprocess_bp = Blueprint('preprocess', __name__, url_prefix='/api/preprocess')

def get_preprocessed_folder():
    backend_dir = os.path.dirname(current_app.root_path)
    folder = os.path.join(backend_dir, 'data', 'preprocessed')
    os.makedirs(folder, exist_ok=True)
    return folder

import threading
from datetime import datetime

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

    # Buat record preprocessing dengan status pending
    # Tentukan nama default
    existing_count = Preprocessing.query.filter_by(dataset_id=dataset_id).count()
    if dataset.dataset_name:
        default_name = dataset.dataset_name
    else:
        default_name = f"Preprocess #{existing_count + 1}"

    # CEK apakah dataset ini sudah pernah dipreprocess (status completed)
    existing_preproc = Preprocessing.query.filter_by(dataset_id=dataset_id, status='completed').first()
    if existing_preproc:
        return jsonify({'error': f'Dataset "{dataset.dataset_name or dataset.filename}" sudah pernah dibersihkan. Tidak boleh memproses ulang.'}), 400

    preprocessing = Preprocessing(
        dataset_id=dataset_id,
        preprocessed_filepath="",
        row_count=0,
        name=default_name,
        status='pending',
        progress=0
    )
    db.session.add(preprocessing)
    db.session.commit()

    # Jalankan proses preprocessing di background thread
    app = current_app._get_current_object()
    thread = threading.Thread(
        target=run_preprocessing_thread,
        args=(app, preprocessing.id, raw_path, dataset.filename)
    )
    thread.start()

    return jsonify({
        'message': 'Preprocessing dimulai',
        'preprocessed_id': preprocessing.id,
        'name': preprocessing.name
    }), 200

def run_preprocessing_thread(app, preproc_id, raw_path, original_filename):
    with app.app_context():
        preprocessing = Preprocessing.query.get(preproc_id)
        if not preprocessing:
            return

        try:
            # Update status awal
            preprocessing.status = 'running'
            preprocessing.progress = 10
            preprocessing.metrics = {'progress_message': 'Membaca file...'}
            db.session.commit()

            df = pd.read_csv(raw_path)
            if 'kalimat' not in df.columns:
                raise ValueError("Kolom 'kalimat' tidak ditemukan")

            total_rows = len(df)
            preprocessing.progress = 30
            preprocessing.metrics = {'progress_message': 'Membersihkan teks...'}
            db.session.commit()

            # Proses per batch agar progress bisa diupdate (simulasi progres)
            cleaned = []
            for i, text in enumerate(df['kalimat']):
                cleaned.append(preprocess_light(str(text)))
                # Update progress setiap 10% atau setiap 500 baris
                if i % max(1, total_rows // 10) == 0:
                    progress = 30 + int((i / total_rows) * 60)  # 30-90%
                    preprocessing.progress = progress
                    preprocessing.row_count = i + 1
                    preprocessing.metrics = {
                        'progress_message': f'Memproses baris {i+1}/{total_rows}'
                    }
                    db.session.commit()

            df['cleaned_kalimat'] = cleaned

            # Simpan file
            preprocessing.progress = 95
            preprocessing.metrics = {'progress_message': 'Menyimpan file...'}
            db.session.commit()

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_name = os.path.splitext(original_filename)[0]
            preprocessed_filename = f"preprocessed_{timestamp}_{original_name}.csv"
            preprocessed_folder = get_preprocessed_folder()
            preprocessed_path = os.path.join(preprocessed_folder, preprocessed_filename)
            df.to_csv(preprocessed_path, index=False)

            preprocessing.preprocessed_filepath = preprocessed_path
            preprocessing.row_count = total_rows
            preprocessing.status = 'completed'
            preprocessing.progress = 100
            preprocessing.metrics = {'progress_message': 'Selesai'}
            db.session.commit()

        except Exception as e:
            preprocessing.status = 'failed'
            preprocessing.progress = 0
            preprocessing.metrics = {'error': str(e), 'progress_message': f'Gagal: {str(e)}'}
            db.session.commit()
            raise e


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

@preprocess_bp.route('/download-heavy/<int:preprocess_id>', methods=['GET', 'OPTIONS'])
@cross_origin(origins=["http://localhost:5500", "http://127.0.0.1:5500"], supports_credentials=True)
def download_heavy(preprocess_id):
    if request.method == 'OPTIONS':
        return '', 200

    preprocessing = Preprocessing.query.get(preprocess_id)
    if not preprocessing:
        return jsonify({'error': 'Preprocessing record not found'}), 404

    # Ambil dataset asli yang terkait
    dataset = Dataset.query.get(preprocessing.dataset_id)
    if not dataset or not os.path.exists(dataset.filepath):
        return jsonify({'error': 'Original dataset file not found'}), 404

    # Baca file mentah
    df = pd.read_csv(dataset.filepath)
    if 'kalimat' not in df.columns:
        return jsonify({'error': "Column 'kalimat' not found in original dataset"}), 400

    # Terapkan preprocessing heavy (stopword + stemming)
    from app.utils.preprocessing_utils import preprocess_heavy
    df['cleaned_kalimat'] = df['kalimat'].astype(str).apply(preprocess_heavy)

    # Simpan ke file sementara
    import tempfile
    fd, temp_path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    df.to_csv(temp_path, index=False)

    # Kirim file, lalu hapus setelah request selesai
    from flask import after_this_request
    @after_this_request
    def remove_file(response):
        try:
            os.remove(temp_path)
        except Exception as e:
            current_app.logger.warning(f"Gagal hapus temp file {temp_path}: {e}")
        return response

    # Nama file download: heavy_{original_filename}
    original_filename = dataset.filename or f"dataset_{dataset.id}"
    download_name = f"preprocessed_heavy_{original_filename}"
    return send_file(temp_path, as_attachment=True, download_name=download_name, mimetype='text/csv')