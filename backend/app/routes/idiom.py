import pandas as pd
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app import db
from app.models.idiom import Idiom
from sqlalchemy import or_, func

idiom_bp = Blueprint('idiom', __name__, url_prefix='/api/idioms')

# GET semua idiom (dengan filter, sort, pagination)
@idiom_bp.route('/', methods=['GET'], strict_slashes=False)
@jwt_required()
def get_idioms():
    search = request.args.get('search', '').strip()
    sort = request.args.get('sort', 'asc')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 5))
    
    query = Idiom.query
    if search:
        query = query.filter(
            or_(
                Idiom.idiom_text.ilike(f'%{search}%'),
                Idiom.idiom_meaning.ilike(f'%{search}%')
            )
        )
    
    if sort == 'asc':
        query = query.order_by(Idiom.idiom_text.asc())
    else:
        query = query.order_by(Idiom.idiom_text.desc())
    
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'data': [{
            'id': i.id,
            'idiom': i.idiom_text,
            'meaning': i.idiom_meaning
        } for i in paginated.items],
        'total': paginated.total,
        'page': page,
        'per_page': per_page,
        'total_pages': paginated.pages
    }), 200

# Tambah idiom manual
@idiom_bp.route('/', methods=['POST'], strict_slashes=False)
@jwt_required()
def add_idiom():
    data = request.get_json()
    idiom_text = data.get('idiom_text', '').strip()
    idiom_meaning = data.get('idiom_meaning', '').strip()
    
    if not idiom_text or not idiom_meaning:
        return jsonify({'message': 'Idiom dan makna harus diisi'}), 400
    
    # Cek kombinasi idiom_text dan idiom_meaning
    existing = Idiom.query.filter(
        func.lower(Idiom.idiom_text) == idiom_text.lower(),
        func.lower(Idiom.idiom_meaning) == idiom_meaning.lower()
    ).first()
    if existing:
        return jsonify({'message': 'Kombinasi idiom dan makna ini sudah ada'}), 409
    
    new_idiom = Idiom(
        idiom_text=idiom_text,
        idiom_meaning=idiom_meaning,
        source='manual'
    )
    db.session.add(new_idiom)
    db.session.commit()
    
    return jsonify({
        'id': new_idiom.id,
        'idiom': new_idiom.idiom_text,
        'meaning': new_idiom.idiom_meaning
    }), 201

# Edit idiom
@idiom_bp.route('/<int:idiom_id>', methods=['PUT'], strict_slashes=False)
@jwt_required()
def update_idiom(idiom_id):
    idiom = Idiom.query.get(idiom_id)
    if not idiom:
        return jsonify({'message': 'Idiom tidak ditemukan'}), 404
    
    data = request.get_json()
    idiom_text = data.get('idiom_text', '').strip()
    idiom_meaning = data.get('idiom_meaning', '').strip()
    
    if not idiom_text or not idiom_meaning:
        return jsonify({'message': 'Idiom dan makna harus diisi'}), 400
    
    # Cek kombinasi idiom_text dan idiom_meaning selain dirinya sendiri
    existing = Idiom.query.filter(
        func.lower(Idiom.idiom_text) == idiom_text.lower(),
        func.lower(Idiom.idiom_meaning) == idiom_meaning.lower(),
        Idiom.id != idiom_id
    ).first()
    if existing:
        return jsonify({'message': 'Kombinasi idiom dan makna ini sudah ada'}), 409
    
    idiom.idiom_text = idiom_text
    idiom.idiom_meaning = idiom_meaning
    db.session.commit()
    
    return jsonify({
        'id': idiom.id,
        'idiom': idiom.idiom_text,
        'meaning': idiom.idiom_meaning
    }), 200

# Hapus idiom
@idiom_bp.route('/<int:idiom_id>', methods=['DELETE'], strict_slashes=False)
@jwt_required()
def delete_idiom(idiom_id):
    idiom = Idiom.query.get(idiom_id)
    if not idiom:
        return jsonify({'message': 'Idiom tidak ditemukan'}), 404
    
    db.session.delete(idiom)
    db.session.commit()
    return jsonify({'message': 'Idiom berhasil dihapus'}), 200

# Upload CSV
@idiom_bp.route('/upload', methods=['POST'], strict_slashes=False)
@jwt_required()
def upload_idioms():
    if 'file' not in request.files:
        return jsonify({'message': 'File tidak ditemukan'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'File tidak dipilih'}), 400
    if not file.filename.endswith('.csv'):
        return jsonify({'message': 'Hanya file CSV yang diperbolehkan'}), 400
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'message': f'Error membaca file: {str(e)}'}), 500
    
    # Cari kolom idiom dan makna
    idiom_col = None
    meaning_col = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['idiom_text', 'idiom']:
            idiom_col = col
        if col_lower in ['idiom_meaning', 'meaning', 'arti']:
            meaning_col = col
    
    if not idiom_col or not meaning_col:
        return jsonify({'message': 'Kolom idiom dan makna tidak ditemukan. Gunakan kolom: idiom_text/idiom dan idiom_meaning/meaning/arti'}), 400
    
    added = 0
    skipped = 0
    for _, row in df.iterrows():
        idiom_text = str(row[idiom_col]).strip()
        idiom_meaning = str(row[meaning_col]).strip()
        
        if not idiom_text or not idiom_meaning or idiom_text == 'nan' or idiom_meaning == 'nan':
            skipped += 1
            continue
        
        # Cek kombinasi idiom_text dan idiom_meaning
        existing = Idiom.query.filter(
            func.lower(Idiom.idiom_text) == idiom_text.lower(),
            func.lower(Idiom.idiom_meaning) == idiom_meaning.lower()
        ).first()
        if not existing:
            new_idiom = Idiom(
                idiom_text=idiom_text,
                idiom_meaning=idiom_meaning,
                source='upload'
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

# Batch POST (untuk preview & ekstrak dari frontend)
@idiom_bp.route('/batch', methods=['POST'], strict_slashes=False)
@jwt_required()
def batch_add_idioms():
    data = request.get_json()
    idioms = data.get('idioms', [])
    if not idioms:
        return jsonify({'message': 'Tidak ada data idiom'}), 400
    
    added = 0
    merged = 0
    for item in idioms:
        idiom_text = item.get('idiom_text', '').strip().lower()
        idiom_meaning = item.get('idiom_meaning', '').strip()
        if not idiom_text or not idiom_meaning:
            continue
        
        # Cek apakah idiom sudah ada (case-insensitive)
        existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == idiom_text).first()
        if existing:
            # Gabungkan arti jika berbeda (pisahkan dengan '; ')
            current_meanings = set([m.strip() for m in existing.idiom_meaning.split(';')])
            if idiom_meaning not in current_meanings:
                existing.idiom_meaning += '; ' + idiom_meaning
                merged += 1
        else:
            new_idiom = Idiom(
                idiom_text=idiom_text,
                idiom_meaning=idiom_meaning,
                source='batch'
            )
            db.session.add(new_idiom)
            added += 1
    
    db.session.commit()
    return jsonify({
        'message': f'Berhasil menambahkan {added} idiom baru, menggabungkan {merged} arti',
        'added': added,
        'merged': merged
    }), 200