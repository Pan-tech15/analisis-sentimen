import csv
import pandas as pd
from flask import Blueprint, request, jsonify, make_response
from flask_jwt_extended import jwt_required
from app import db
from app.models.idiom import Idiom
from io import StringIO
from sqlalchemy import or_, func

idiom_bp = Blueprint('idiom', __name__, url_prefix='/api/idioms')

# Helper: gabungkan arti (lowercase, unique, sorted)
def merge_meanings(existing_meaning, new_meaning):
    existing_set = set([m.strip().lower() for m in existing_meaning.split(';')])
    new_meaning_lower = new_meaning.strip().lower()
    if new_meaning_lower and new_meaning_lower not in existing_set:
        existing_set.add(new_meaning_lower)
        return '; '.join(sorted(existing_set))
    return existing_meaning

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
            'meaning': i.idiom_meaning,
            'emotion': i.emotion
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
    idiom_text = data.get('idiom_text', '').strip().lower()
    idiom_meaning = data.get('idiom_meaning', '').strip().lower()
    emotion = data.get('emotion', '').strip().lower() or None

    if not idiom_text or not idiom_meaning:
        return jsonify({'message': 'Idiom dan makna harus diisi'}), 400

    existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == idiom_text).first()
    if existing:
        new_meaning = merge_meanings(existing.idiom_meaning, idiom_meaning)
        if new_meaning != existing.idiom_meaning:
            existing.idiom_meaning = new_meaning
        if emotion and existing.emotion != emotion:
            existing.emotion = emotion
        db.session.commit()
        return jsonify({
            'id': existing.id,
            'idiom': existing.idiom_text,
            'meaning': existing.idiom_meaning,
            'emotion': existing.emotion,
            'message': 'Arti digabungkan'
        }), 200
    else:
        new_idiom = Idiom(
            idiom_text=idiom_text,
            idiom_meaning=idiom_meaning,
            emotion=emotion,
            source='manual'
        )
        db.session.add(new_idiom)
        db.session.commit()
        return jsonify({
            'id': new_idiom.id,
            'idiom': new_idiom.idiom_text,
            'meaning': new_idiom.idiom_meaning,
            'emotion': new_idiom.emotion
        }), 201

# Edit idiom
@idiom_bp.route('/<int:idiom_id>', methods=['PUT'], strict_slashes=False)
@jwt_required()
def update_idiom(idiom_id):
    idiom = Idiom.query.get(idiom_id)
    if not idiom:
        return jsonify({'message': 'Idiom tidak ditemukan'}), 404
    
    data = request.get_json()
    new_text = data.get('idiom_text', '').strip().lower()
    new_meaning = data.get('idiom_meaning', '').strip().lower()
    emotion = data.get('emotion', '').strip().lower() or None
    
    if not new_text or not new_meaning:
        return jsonify({'message': 'Idiom dan makna harus diisi'}), 400
    
    # Jika teks idiom berubah
    if new_text != idiom.idiom_text:
        existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == new_text).first()
        if existing:
            # Gabungkan arti ke idiom yang sudah ada, lalu hapus idiom lama
            merged_meaning = merge_meanings(existing.idiom_meaning, new_meaning)
            existing.idiom_meaning = merged_meaning
            if emotion and existing.emotion != emotion:
                existing.emotion = emotion
            db.session.delete(idiom)
            db.session.commit()
            return jsonify({
                'id': existing.id,
                'idiom': existing.idiom_text,
                'meaning': existing.idiom_meaning,
                'emotion': existing.emotion,
                'message': 'Idiom digabung dengan yang sudah ada'
            }), 200
        else:
            idiom.idiom_text = new_text
            idiom.idiom_meaning = new_meaning
            idiom.emotion = emotion
    else:
        # Teks sama, update arti (timpa, tidak digabung karena edit biasanya penggantian total)
        idiom.idiom_meaning = new_meaning
        idiom.emotion = emotion
    
    db.session.commit()
    return jsonify({
        'id': idiom.id,
        'idiom': idiom.idiom_text,
        'meaning': idiom.idiom_meaning,
        'emotion': idiom.emotion
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
    
    # Deteksi kolom
    idiom_col = None
    meaning_col = None
    emotion_col = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['idiom_text', 'idiom']:
            idiom_col = col
        if col_lower in ['idiom_meaning', 'meaning', 'arti']:
            meaning_col = col
        if col_lower in ['emotion', 'emosi', 'label', 'sentimen']:
            emotion_col = col
    
    if not idiom_col or not meaning_col:
        return jsonify({'message': 'Kolom idiom dan makna tidak ditemukan.'}), 400
    
    added = 0
    merged = 0
    skipped = 0
    for _, row in df.iterrows():
        idiom_text = str(row[idiom_col]).strip().lower()
        idiom_meaning = str(row[meaning_col]).strip().lower()
        emotion = None
        if emotion_col:
            raw_emotion = str(row[emotion_col]).strip().lower()
            if raw_emotion and raw_emotion != 'nan':
                emotion = raw_emotion
        
        if not idiom_text or not idiom_meaning or idiom_text == 'nan' or idiom_meaning == 'nan':
            skipped += 1
            continue
        
        existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == idiom_text).first()
        if existing:
            new_meaning = merge_meanings(existing.idiom_meaning, idiom_meaning)
            if new_meaning != existing.idiom_meaning:
                existing.idiom_meaning = new_meaning
                merged += 1
            if emotion and existing.emotion != emotion:
                existing.emotion = emotion
        else:
            new_idiom = Idiom(
                idiom_text=idiom_text,
                idiom_meaning=idiom_meaning,
                emotion=emotion,
                source='upload'
            )
            db.session.add(new_idiom)
            added += 1
    
    db.session.commit()
    return jsonify({
        'message': f'Berhasil menambahkan {added} idiom baru, menggabungkan {merged} arti, {skipped} baris tidak valid',
        'added': added,
        'merged': merged,
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
        idiom_meaning = item.get('idiom_meaning', '').strip().lower()
        emotion = item.get('emotion', '').strip().lower() or None
        
        if not idiom_text or not idiom_meaning:
            continue
        
        existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == idiom_text).first()
        if existing:
            new_meaning = merge_meanings(existing.idiom_meaning, idiom_meaning)
            if new_meaning != existing.idiom_meaning:
                existing.idiom_meaning = new_meaning
                merged += 1
            if emotion and existing.emotion != emotion:
                existing.emotion = emotion
        else:
            new_idiom = Idiom(
                idiom_text=idiom_text,
                idiom_meaning=idiom_meaning,
                emotion=emotion,
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

# Export semua idiom ke CSV
@idiom_bp.route('/export', methods=['GET'], strict_slashes=False)
@jwt_required()
def export_idioms():
    all_idioms = Idiom.query.order_by(Idiom.idiom_text.asc()).all()
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['idiom_text', 'idiom_meaning', 'emotion'])
    for idiom in all_idioms:
        emotion = idiom.emotion if idiom.emotion else '-'
        writer.writerow([idiom.idiom_text, idiom.idiom_meaning, emotion])
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=idioms_export.csv'
    response.headers['Content-type'] = 'text/csv'
    return response