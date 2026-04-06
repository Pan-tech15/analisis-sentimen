import os
import pandas as pd
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from app import db
from app.models.dataset import Dataset as DatasetModel
from app.models.user import User
from datetime import datetime

dataset_bp = Blueprint('dataset', __name__, url_prefix='/api/datasets')

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@dataset_bp.route('/upload', methods=['POST'])
@jwt_required()
def upload_dataset():
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'message': 'User not found'}), 401

    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'message': 'Only CSV files allowed'}), 400

    filename = secure_filename(file.filename)
    # Buat nama unik dengan timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_filename = f"{timestamp}_{filename}"
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/raw')
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, saved_filename)
    file.save(filepath)

    # Baca CSV untuk menghitung baris dan statistik
    try:
        df = pd.read_csv(filepath)
        required_columns = {'id', 'kalimat', 'has_idiom', 'idiom_text', 'emotion', 'idiom_meaning'}
        if not required_columns.issubset(df.columns):
            os.remove(filepath)
            return jsonify({'message': f'Missing columns. Required: {required_columns}'}), 400

        row_count = len(df)
        has_idiom_count = int(df['has_idiom'].sum()) if 'has_idiom' in df else 0
        no_idiom_count = row_count - has_idiom_count
    except Exception as e:
        os.remove(filepath)
        return jsonify({'message': f'Error reading CSV: {str(e)}'}), 400

    # Simpan ke database
    new_dataset = DatasetModel(
        filename=saved_filename,
        filepath=filepath,
        row_count=row_count,
        has_idiom_count=has_idiom_count,
        no_idiom_count=no_idiom_count,
        uploaded_by=current_user_id
    )
    db.session.add(new_dataset)
    db.session.commit()

    return jsonify({
        'message': 'File uploaded successfully',
        'dataset': {
            'id': new_dataset.id,
            'filename': new_dataset.filename,
            'rows': new_dataset.row_count,
            'has_idiom': new_dataset.has_idiom_count,
            'no_idiom': new_dataset.no_idiom_count,
            'uploaded_at': new_dataset.uploaded_at.isoformat()
        }
    }), 201


@dataset_bp.route('/', methods=['GET'])
@jwt_required()
def list_datasets():
    datasets = DatasetModel.query.order_by(DatasetModel.uploaded_at.desc()).all()
    result = []
    for ds in datasets:
        result.append({
            'id': ds.id,
            'filename': ds.filename,
            'rows': ds.row_count,
            'has_idiom': ds.has_idiom_count,
            'no_idiom': ds.no_idiom_count,
            'uploaded_at': ds.uploaded_at.isoformat()
        })
    return jsonify(result), 200


@dataset_bp.route('/<int:dataset_id>', methods=['DELETE'])
@jwt_required()
def delete_dataset(dataset_id):
    dataset = DatasetModel.query.get(dataset_id)
    if not dataset:
        return jsonify({'message': 'Dataset not found'}), 404
    
    # Hapus file fisik
    if os.path.exists(dataset.filepath):
        os.remove(dataset.filepath)
    
    db.session.delete(dataset)
    db.session.commit()
    return jsonify({'message': 'Dataset deleted'}), 200


@dataset_bp.route('/<int:dataset_id>/preview', methods=['GET'])
@jwt_required()
def preview_dataset(dataset_id):
    dataset = DatasetModel.query.get(dataset_id)
    if not dataset:
        return jsonify({'message': 'Dataset not found'}), 404
    
    # Baca file CSV
    try:
        df = pd.read_csv(dataset.filepath)
    except Exception as e:
        return jsonify({'message': f'Error reading file: {str(e)}'}), 500
    
    # Parameter filter & pagination
    search = request.args.get('search', '').lower()
    emotion = request.args.get('emotion', '')
    has_idiom = request.args.get('has_idiom', '')  # '1', '0', atau ''
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    # Filter data
    filtered_df = df.copy()
    if search:
        filtered_df = filtered_df[filtered_df['kalimat'].str.lower().str.contains(search, na=False)]
    if emotion:
        filtered_df = filtered_df[filtered_df['emotion'].str.lower() == emotion.lower()]
    if has_idiom != '':
        filtered_df = filtered_df[filtered_df['has_idiom'] == int(has_idiom)]
    
    total_rows = len(filtered_df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = filtered_df.iloc[start:end]
    
    records = page_df.to_dict(orient='records')
    # Konversi NaN ke None
    for rec in records:
        for k, v in rec.items():
            if pd.isna(v):
                rec[k] = None
    
    return jsonify({
        'dataset_id': dataset_id,
        'filename': dataset.filename,
        'total_rows': total_rows,
        'page': page,
        'per_page': per_page,
        'data': records
    }), 200