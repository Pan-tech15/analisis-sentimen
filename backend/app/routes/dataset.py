import os
import pandas as pd
from flask import Blueprint, request, jsonify, current_app, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from app import db
from app.models.dataset import Dataset as DatasetModel
from app.models.user import User
from app.models.idiom import Idiom
from datetime import datetime

dataset_bp = Blueprint('dataset', __name__, url_prefix='/api/datasets')

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------- UPLOAD DATASET -------------------
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

    # Ambil dataset_name dari form
    dataset_name = request.form.get('dataset_name', '').strip()
    if not dataset_name:
        dataset_name = None

    original_filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_filename = f"{timestamp}_{original_filename}"

    # PERBAIKAN: gunakan string 'UPLOAD_FOLDER', bukan variabel upload_folder
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/raw')
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, saved_filename)
    file.save(filepath)

    # Baca CSV untuk statistik
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

    new_dataset = DatasetModel(
        filename=saved_filename,
        original_filename=original_filename,
        dataset_name=dataset_name,
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
            'original_filename': new_dataset.original_filename,
            'dataset_name': new_dataset.dataset_name,
            'rows': new_dataset.row_count,
            'has_idiom': new_dataset.has_idiom_count,
            'no_idiom': new_dataset.no_idiom_count,
            'uploaded_at': new_dataset.uploaded_at.isoformat()
        }
    }), 201


# ------------------- LIST DATASETS -------------------
@dataset_bp.route('/', methods=['GET'])
@jwt_required()
def list_datasets():
    datasets = DatasetModel.query.order_by(DatasetModel.uploaded_at.desc()).all()
    result = []
    for ds in datasets:
        result.append({
            'id': ds.id,
            'filename': ds.filename,
            'original_filename': ds.original_filename,
            'dataset_name': ds.dataset_name,
            'rows': ds.row_count,
            'has_idiom': ds.has_idiom_count,
            'no_idiom': ds.no_idiom_count,
            'uploaded_at': ds.uploaded_at.isoformat()
        })
    return jsonify(result), 200


# ------------------- RENAME DATASET -------------------
@dataset_bp.route('/<int:dataset_id>', methods=['PUT'])
@jwt_required()
def rename_dataset(dataset_id):
    dataset = DatasetModel.query.get(dataset_id)
    if not dataset:
        return jsonify({'message': 'Dataset not found'}), 404

    data = request.get_json()
    new_name = data.get('dataset_name', '').strip()
    if not new_name:
        return jsonify({'message': 'Dataset name cannot be empty'}), 400

    dataset.dataset_name = new_name
    db.session.commit()

    return jsonify({
        'message': 'Dataset renamed successfully',
        'dataset': {
            'id': dataset.id,
            'dataset_name': dataset.dataset_name,
            'original_filename': dataset.original_filename,
        }
    }), 200


# ------------------- DELETE DATASET -------------------
@dataset_bp.route('/<int:dataset_id>', methods=['DELETE'])
@jwt_required()
def delete_dataset(dataset_id):
    dataset = DatasetModel.query.get(dataset_id)
    if not dataset:
        return jsonify({'message': 'Dataset not found'}), 404

    if os.path.exists(dataset.filepath):
        os.remove(dataset.filepath)

    db.session.delete(dataset)
    db.session.commit()
    return jsonify({'message': 'Dataset deleted'}), 200


# ------------------- DOWNLOAD DATASET -------------------
@dataset_bp.route('/<int:dataset_id>/download', methods=['GET'])
@jwt_required()
def download_dataset(dataset_id):
    dataset = DatasetModel.query.get(dataset_id)
    if not dataset:
        return jsonify({'message': 'Dataset not found'}), 404

    filepath = dataset.filepath
    # Jika path masih relatif, jadikan absolut
    if not os.path.isabs(filepath):
        upload_folder = os.path.abspath(current_app.config.get('UPLOAD_FOLDER', 'data/raw'))
        filepath = os.path.join(upload_folder, os.path.basename(filepath))

    if not os.path.exists(filepath):
        return jsonify({'message': 'File not found on server'}), 404

    download_name = dataset.original_filename or dataset.filename
    return send_file(filepath, as_attachment=True, download_name=download_name)


# ------------------- EXTRACT IDIOMS -------------------
@dataset_bp.route('/<int:dataset_id>/extract-idioms', methods=['POST'])
@jwt_required()
def extract_idioms_to_kamus(dataset_id):
    try:
        dataset = DatasetModel.query.get(dataset_id)
        if not dataset:
            return jsonify({'message': 'Dataset not found'}), 404

        df = pd.read_csv(dataset.filepath)

        if 'has_idiom' not in df.columns:
            return jsonify({'message': 'Kolom has_idiom tidak ditemukan'}), 400
        if 'idiom_text' not in df.columns or 'idiom_meaning' not in df.columns:
            return jsonify({'message': 'Kolom idiom_text atau idiom_meaning tidak ditemukan'}), 400

        idiom_df = df[df['has_idiom'] == 1]
        if idiom_df.empty:
            return jsonify({'message': 'Tidak ada data idiom dalam dataset ini', 'added': 0, 'skipped': 0}), 200

        added = 0
        skipped = 0

        for _, row in idiom_df.iterrows():
            idiom_text = str(row['idiom_text']).strip()
            idiom_meaning = str(row['idiom_meaning']).strip()
            if not idiom_text or not idiom_meaning:
                skipped += 1
                continue

            existing = Idiom.query.filter(
                db.func.lower(Idiom.idiom_text) == idiom_text.lower(),
                db.func.lower(Idiom.idiom_meaning) == idiom_meaning.lower()
            ).first()

            if not existing:
                new_idiom = Idiom(
                    idiom_text=idiom_text,
                    idiom_meaning=idiom_meaning,
                    source=f'dataset_{dataset_id}'
                )
                db.session.add(new_idiom)
                added += 1
            else:
                skipped += 1

        db.session.commit()
        return jsonify({
            'message': f'Berhasil menambahkan {added} idiom baru, {skipped} idiom sudah ada/tidak valid',
            'added': added,
            'skipped': skipped
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Terjadi kesalahan: {str(e)}'}), 500


# ------------------- PREVIEW DATASET -------------------
@dataset_bp.route('/<int:dataset_id>/preview', methods=['GET'])
@jwt_required()
def preview_dataset(dataset_id):
    dataset = DatasetModel.query.get(dataset_id)
    if not dataset:
        return jsonify({'message': 'Dataset not found'}), 404

    try:
        df = pd.read_csv(dataset.filepath)
    except Exception as e:
        return jsonify({'message': f'Error reading file: {str(e)}'}), 500

    search = request.args.get('search', '').lower()
    emotion = request.args.get('emotion', '')
    has_idiom = request.args.get('has_idiom', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))

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