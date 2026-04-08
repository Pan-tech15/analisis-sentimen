import os
import pandas as pd
from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models.dataset import Dataset as DatasetModel
from app.models.preprocessing import Preprocessing
from app.utils.preprocessing_utils import preprocess_text
from datetime import datetime
import tempfile

preprocess_bp = Blueprint('preprocess', __name__, url_prefix='/api/preprocess')

@preprocess_bp.route('/start', methods=['POST'])
@jwt_required()
def start_preprocessing():
    data = request.get_json()
    dataset_id = data.get('dataset_id')
    if not dataset_id:
        return jsonify({'message': 'Dataset ID diperlukan'}), 400
    
    dataset = DatasetModel.query.get(dataset_id)
    if not dataset:
        return jsonify({'message': 'Dataset tidak ditemukan'}), 404
    
    # Baca file CSV
    try:
        df = pd.read_csv(dataset.filepath)
    except Exception as e:
        return jsonify({'message': f'Error membaca file: {str(e)}'}), 500
    
    # Kolom yang diperlukan: 'kalimat' dan 'emotion' (atau 'label')
    if 'kalimat' not in df.columns:
        return jsonify({'message': 'Kolom "kalimat" tidak ditemukan'}), 400
    
    # Asumsikan label emosi ada di kolom 'emotion'
    emotion_col = 'emotion' if 'emotion' in df.columns else 'label' if 'label' in df.columns else None
    if not emotion_col:
        return jsonify({'message': 'Kolom emosi/label tidak ditemukan (gunakan "emotion" atau "label")'}), 400
    
    # Proses setiap kalimat
    results = []
    total = len(df)
    for idx, row in df.iterrows():
        original = row['kalimat']
        cleaned = preprocess_text(original)
        results.append({
            'original': original,
            'cleaned': cleaned,
            'emotion': row[emotion_col]
        })
    
    # Simpan hasil ke CSV
    result_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    preprocessed_filename = f"preprocessed_{timestamp}_{dataset.filename}"
    preprocessed_path = os.path.join('data', 'preprocessed', preprocessed_filename)
    os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
    result_df.to_csv(preprocessed_path, index=False)
    
    # Simpan record preprocessing ke database
    preproc_record = Preprocessing(
        dataset_id=dataset_id,
        preprocessed_filepath=preprocessed_path,
        row_count=total
    )
    db.session.add(preproc_record)
    db.session.commit()
    
    return jsonify({
        'message': 'Preprocessing selesai',
        'preprocessed_id': preproc_record.id,
        'file_url': f'/api/preprocess/download/{preproc_record.id}',
        'rows': total
    }), 200

@preprocess_bp.route('/download/<int:preproc_id>', methods=['GET'])
@jwt_required()
def download_preprocessed(preproc_id):
    preproc = Preprocessing.query.get(preproc_id)
    if not preproc:
        return jsonify({'message': 'Record tidak ditemukan'}), 404
    
    if not os.path.exists(preproc.preprocessed_filepath):
        return jsonify({'message': 'File tidak ditemukan'}), 404
    
    return send_file(
        preproc.preprocessed_filepath,
        as_attachment=True,
        download_name=f"preprocessed_{preproc.id}.csv"
    )