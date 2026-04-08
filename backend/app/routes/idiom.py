import pandas as pd
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app import db
from app.models.idiom import Idiom
from sqlalchemy import or_, func

idiom_bp = Blueprint('idiom', __name__, url_prefix='/api/idioms')

def normalize_text(text):
    return text.strip().lower() if text else ''

def merge_meanings(existing_meaning, new_meaning):
    existing_set = set([m.strip() for m in existing_meaning.split(';')]) if existing_meaning else set()
    new_meaning_clean = new_meaning.strip()
    if new_meaning_clean not in existing_set:
        if existing_meaning:
            return existing_meaning + '; ' + new_meaning_clean
        else:
            return new_meaning_clean
    return existing_meaning

@idiom_bp.route('/', methods=['GET'], strict_slashes=False)
@jwt_required()
def get_idioms():
    search = request.args.get('search', '').strip().lower()
    sort = request.args.get('sort', 'asc')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
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

@idiom_bp.route('/', methods=['POST'], strict_slashes=False)
@jwt_required()
def add_idiom():
    data = request.get_json()
    idiom_text = normalize_text(data.get('idiom_text', ''))
    idiom_meaning = data.get('idiom_meaning', '').strip()
    
    if not idiom_text or not idiom_meaning:
        return jsonify({'message': 'Idiom dan makna harus diisi'}), 400
    
    existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == idiom_text).first()
    if existing:
        new_meaning = merge_meanings(existing.idiom_meaning, idiom_meaning)
        existing.idiom_meaning = new_meaning
        db.session.commit()
        return jsonify({
            'id': existing.id,
            'idiom': existing.idiom_text,
            'meaning': existing.idiom_meaning,
            'message': 'Arti idiom ditambahkan ke yang sudah ada'
        }), 200
    else:
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

@idiom_bp.route('/<int:idiom_id>', methods=['PUT'], strict_slashes=False)
@jwt_required()
def update_idiom(idiom_id):
    idiom = Idiom.query.get(idiom_id)
    if not idiom:
        return jsonify({'message': 'Idiom tidak ditemukan'}), 404
    
    data = request.get_json()
    new_text = normalize_text(data.get('idiom_text', ''))
    new_meaning = data.get('idiom_meaning', '').strip()
    
    if not new_text or not new_meaning:
        return jsonify({'message': 'Idiom dan makna harus diisi'}), 400
    
    if new_text != idiom.idiom_text:
        existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == new_text).first()
        if existing:
            return jsonify({'message': 'Idiom dengan teks tersebut sudah ada'}), 409
    
    idiom.idiom_text = new_text
    idiom.idiom_meaning = new_meaning
    db.session.commit()
    return jsonify({
        'id': idiom.id,
        'idiom': idiom.idiom_text,
        'meaning': idiom.idiom_meaning
    }), 200

@idiom_bp.route('/<int:idiom_id>', methods=['DELETE'], strict_slashes=False)
@jwt_required()
def delete_idiom(idiom_id):
    idiom = Idiom.query.get(idiom_id)
    if not idiom:
        return jsonify({'message': 'Idiom tidak ditemukan'}), 404
    db.session.delete(idiom)
    db.session.commit()
    return jsonify({'message': 'Idiom berhasil dihapus'}), 200

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
        idiom_text = normalize_text(item.get('idiom_text', ''))
        idiom_meaning = item.get('idiom_meaning', '').strip()
        if not idiom_text or not idiom_meaning:
            continue
        
        existing = Idiom.query.filter(func.lower(Idiom.idiom_text) == idiom_text).first()
        if existing:
            new_meaning = merge_meanings(existing.idiom_meaning, idiom_meaning)
            if new_meaning != existing.idiom_meaning:
                existing.idiom_meaning = new_meaning
                merged += 1
        else:
            new_idiom = Idiom(
                idiom_text=idiom_text,
                idiom_meaning=idiom_meaning,
                source='upload_batch'
            )
            db.session.add(new_idiom)
            added += 1
    
    db.session.commit()
    return jsonify({
        'message': f'Berhasil menambahkan {added} idiom baru, menggabungkan {merged} arti ke idiom yang sudah ada',
        'added': added,
        'merged': merged
    }), 200