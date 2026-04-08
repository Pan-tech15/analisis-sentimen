import os
import pandas as pd
from flask import Blueprint, request, jsonify, send_file, current_app
from flask_jwt_extended import jwt_required
from app import db
from app.models.dataset import Dataset as DatasetModel
from app.models.preprocessing import Preprocessing
from app.utils.preprocessing_utils import preprocess_text
from datetime import datetime

preprocess_bp = Blueprint('preprocess', __name__, url_prefix='/api/preprocess')

def get_preprocessed_folder():
    """Mendapatkan path absolut folder data/preprocessed"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # sampai folder backend
    preprocessed_dir = os.path.join(base_dir, 'data', 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)
    return preprocessed_dir

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
    
    try:
        df = pd.read_csv(dataset.filepath)
    except Exception as e:
        return jsonify({'message': f'Error membaca file: {str(e)}'}), 500
    
    if 'kalimat' not in df.columns:
        return jsonify({'message': 'Kolom "kalimat" tidak ditemukan'}), 400
    
    emotion_col = 'emotion' if 'emotion' in df.columns else 'label' if 'label' in df.columns else None
    if not emotion_col:
        return jsonify({'message': 'Kolom emosi/label tidak ditemukan (gunakan "emotion" atau "label")'}), 400
    
    results = []
    total = len(df)
    for _, row in df.iterrows():
        original = row['kalimat']
        cleaned = preprocess_text(original)
        results.append({
            'original': original,
            'cleaned': cleaned,
            'emotion': row[emotion_col]
        })
    
    result_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_filename = os.path.basename(dataset.filename)
    preprocessed_filename = f"preprocessed_{timestamp}_{original_filename}"
    preprocessed_dir = get_preprocessed_folder()
    preprocessed_path = os.path.join(preprocessed_dir, preprocessed_filename)
    result_df.to_csv(preprocessed_path, index=False)
    
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