import csv
import pandas as pd
from flask import Blueprint, request, jsonify, make_response
from flask_jwt_extended import jwt_required
from app import db
from app.models.idiom import Idiom
from io import StringIO
from sqlalchemy import or_, func

idiom_bp = Blueprint('idiom', __name__, url_prefix='/api/idioms')

# ========== MAPPING EMOSI (Indonesia ↔ Inggris) ==========
EMOJI_MAP = {
    'senang': 'Happy', 'happy': 'Happy',
    'sedih': 'Sad', 'sad': 'Sad',
    'marah': 'Angry', 'angry': 'Angry',
    'takut': 'Fear', 'fear': 'Fear',
    'terkejut': 'Surprise', 'surprise': 'Surprise',
    'percaya': 'Trust', 'trust': 'Trust',
    'netral': 'Neutral', 'neutral': 'Neutral'
}

def normalize_emotion(emotion):
    """Ubah emosi ke format Inggris (Happy, Sad, dll)"""
    if not emotion:
        return None
    emotion_lower = emotion.strip().lower()
    return EMOJI_MAP.get(emotion_lower, emotion_lower.capitalize())

# Helper gabung arti (sama seperti sebelumnya)
def merge_meanings(existing_meaning, new_meaning):
    existing_set = set([m.strip().lower() for m in existing_meaning.split(';')])
    new_meaning_lower = new_meaning.strip().lower()
    if new_meaning_lower and new_meaning_lower not in existing_set:
        existing_set.add(new_meaning_lower)
        return '; '.join(sorted(existing_set))
    return existing_meaning

# GET semua idiom (dengan filter, sort, pagination, dan filter emosi)
@idiom_bp.route('/', methods=['GET'], strict_slashes=False)
@jwt_required()
def get_idioms():
    search = request.args.get('search', '').strip()
    emotion_filter = request.args.get('emotion', '').strip()  # filter dalam bahasa Inggris
    sort = request.args.get('sort', 'asc')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 5))
    
    query = Idiom.query
    
    # Filter teks
    if search:
        query = query.filter(
            or_(
                Idiom.idiom_text.ilike(f'%{search}%'),
                Idiom.idiom_meaning.ilike(f'%{search}%')
            )
        )
    
    # Filter emosi (case‑insensitive, karena di database bisa Indonesia/Inggris)
    if emotion_filter:
        possible_values = [emotion_filter]
        # Tambahkan padanan Indonesia jika filter dalam Inggris
        for indo, eng in EMOJI_MAP.items():
            if eng.lower() == emotion_filter.lower():
                possible_values.append(indo)
        # Gunakan ILIKE untuk semua kemungkinan
        emotion_filter_lower = emotion_filter.lower()
        conditions = []
        for val in possible_values:
            conditions.append(Idiom.emotion.ilike(val))
        # Jika tidak ada kondisi, tambahkan kondisi yang selalu false (tidak perlu)
        if conditions:
            from sqlalchemy import or_
            query = query.filter(or_(*conditions))
        else:
            # Fallback ke pencarian langsung (case‑insensitive)
            query = query.filter(Idiom.emotion.ilike(f'%{emotion_filter}%'))
    
    # Sorting
    if sort == 'asc':
        query = query.order_by(Idiom.idiom_text.asc())
    else:
        query = query.order_by(Idiom.idiom_text.desc())
    
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Format output dengan emosi dalam Inggris (gunakan mapping)
    data = []
    for i in paginated.items:
        emotion_norm = normalize_emotion(i.emotion) if i.emotion else None
        data.append({
            'id': i.id,
            'idiom': i.idiom_text,
            'meaning': i.idiom_meaning,
            'emotion': emotion_norm  # selalu Inggris
        })
    
    return jsonify({
        'data': data,
        'total': paginated.total,
        'page': page,
        'per_page': per_page,
        'total_pages': paginated.pages
    }), 200

# Tambah idiom manual (input emosi bisa Indonesia atau Inggris, disimpan Inggris)
@idiom_bp.route('/', methods=['POST'], strict_slashes=False)
@jwt_required()
def add_idiom():
    data = request.get_json()
    idiom_text = data.get('idiom_text', '').strip().lower()
    idiom_meaning = data.get('idiom_meaning', '').strip().lower()
    emotion = data.get('emotion', '').strip()
    emotion_normalized = normalize_emotion(emotion) if emotion else None

    if not idiom_text or not idiom_meaning:
        return jsonify({'message': 'Idiom dan makna harus diisi'}), 400

    existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == idiom_text).first()
    if existing:
        new_meaning = merge_meanings(existing.idiom_meaning, idiom_meaning)
        if new_meaning != existing.idiom_meaning:
            existing.idiom_meaning = new_meaning
        if emotion_normalized and existing.emotion != emotion_normalized:
            existing.emotion = emotion_normalized
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
            emotion=emotion_normalized,
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

# Edit idiom (emosi dinormalisasi)
@idiom_bp.route('/<int:idiom_id>', methods=['PUT'], strict_slashes=False)
@jwt_required()
def update_idiom(idiom_id):
    idiom = Idiom.query.get(idiom_id)
    if not idiom:
        return jsonify({'message': 'Idiom tidak ditemukan'}), 404
    
    data = request.get_json()
    new_text = data.get('idiom_text', '').strip().lower()
    new_meaning = data.get('idiom_meaning', '').strip().lower()
    emotion = data.get('emotion', '').strip()
    emotion_normalized = normalize_emotion(emotion) if emotion else None
    
    if not new_text or not new_meaning:
        return jsonify({'message': 'Idiom dan makna harus diisi'}), 400
    
    if new_text != idiom.idiom_text:
        existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == new_text).first()
        if existing:
            merged_meaning = merge_meanings(existing.idiom_meaning, new_meaning)
            existing.idiom_meaning = merged_meaning
            if emotion_normalized and existing.emotion != emotion_normalized:
                existing.emotion = emotion_normalized
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
            idiom.emotion = emotion_normalized
    else:
        idiom.idiom_meaning = new_meaning
        if emotion_normalized:
            idiom.emotion = emotion_normalized
    
    db.session.commit()
    return jsonify({
        'id': idiom.id,
        'idiom': idiom.idiom_text,
        'meaning': idiom.idiom_meaning,
        'emotion': idiom.emotion
    }), 200

# Hapus idiom (sama)
@idiom_bp.route('/<int:idiom_id>', methods=['DELETE'], strict_slashes=False)
@jwt_required()
def delete_idiom(idiom_id):
    idiom = Idiom.query.get(idiom_id)
    if not idiom:
        return jsonify({'message': 'Idiom tidak ditemukan'}), 404
    db.session.delete(idiom)
    db.session.commit()
    return jsonify({'message': 'Idiom berhasil dihapus'}), 200

# Upload CSV (emosi dinormalisasi)
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
            raw_emotion = str(row[emotion_col]).strip()
            if raw_emotion and raw_emotion != 'nan':
                emotion = normalize_emotion(raw_emotion)
        
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

# Batch POST (emosi dinormalisasi)
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
        emotion = item.get('emotion', '').strip()
        emotion_norm = normalize_emotion(emotion) if emotion else None
        
        if not idiom_text or not idiom_meaning:
            continue
        
        existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == idiom_text).first()
        if existing:
            new_meaning = merge_meanings(existing.idiom_meaning, idiom_meaning)
            if new_meaning != existing.idiom_meaning:
                existing.idiom_meaning = new_meaning
                merged += 1
            if emotion_norm and existing.emotion != emotion_norm:
                existing.emotion = emotion_norm
        else:
            new_idiom = Idiom(
                idiom_text=idiom_text,
                idiom_meaning=idiom_meaning,
                emotion=emotion_norm,
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

# Export semua idiom ke CSV (emosi dalam Inggris)
@idiom_bp.route('/export', methods=['GET'], strict_slashes=False)
@jwt_required()
def export_idioms():
    all_idioms = Idiom.query.order_by(Idiom.idiom_text.asc()).all()
    output = StringIO()
    # Tambahkan BOM UTF-8 agar Excel membaca karakter dengan benar
    output.write('\ufeff')
    writer = csv.writer(output, delimiter=';')
    writer.writerow(['idiom_text', 'idiom_meaning', 'emotion'])
    for idiom in all_idioms:
        emotion = normalize_emotion(idiom.emotion) if idiom.emotion else ''
        writer.writerow([idiom.idiom_text, idiom.idiom_meaning, emotion])
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=idioms_export.csv'
    response.headers['Content-type'] = 'text/csv; charset=utf-8'
    return response