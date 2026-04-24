from flask import Blueprint, request, jsonify
from app import db
from app.models.split_ratio import SplitRatio

split_ratio_bp = Blueprint('split_ratio', __name__, url_prefix='/api/split-ratios')

# GET semua split ratios
@split_ratio_bp.route('', methods=['GET'])
def get_all():
    ratios = SplitRatio.query.order_by(SplitRatio.created_at.desc()).all()
    return jsonify([r.to_dict() for r in ratios]), 200

# GET satu split ratio
@split_ratio_bp.route('/<int:ratio_id>', methods=['GET'])
def get_one(ratio_id):
    ratio = SplitRatio.query.get(ratio_id)
    if not ratio:
        return jsonify({'error': 'Split ratio not found'}), 404
    return jsonify(ratio.to_dict()), 200

# POST create baru
@split_ratio_bp.route('', methods=['POST'])
def create():
    data = request.get_json()
    train = data.get('train')
    test = data.get('test')
    name = data.get('name')             # sekarang opsional

    if train is None or test is None:
        return jsonify({'error': 'train and test required'}), 400
    try:
        train = int(train)
        test = int(test)
    except ValueError:
        return jsonify({'error': 'train dan test harus angka'}), 400
    if train + test != 100:
        return jsonify({'error': 'Total harus 100'}), 400

    # Buat nama otomatis jika tidak diberikan
    if not name:
        name = f"{train}-{test}"

    # Hindari duplikasi nama
    existing = SplitRatio.query.filter_by(name=name).first()
    if existing:
        import uuid
        name = f"{name}_{uuid.uuid4().hex[:4]}"

    ratio = SplitRatio(name=name, train_pct=train, test_pct=test)
    db.session.add(ratio)
    db.session.commit()
    return jsonify(ratio.to_dict()), 201

# PUT ubah ratio
@split_ratio_bp.route('/<int:ratio_id>', methods=['PUT'])
def update(ratio_id):
    ratio = SplitRatio.query.get(ratio_id)
    if not ratio:
        return jsonify({'error': 'Split ratio not found'}), 404

    data = request.get_json()
    name = data.get('name')
    train = data.get('train')
    test = data.get('test')
    if name is not None:
        # cek nama baru tidak bentrok
        exist = SplitRatio.query.filter(SplitRatio.name == name, SplitRatio.id != ratio_id).first()
        if exist:
            return jsonify({'error': 'Nama sudah digunakan'}), 409
        ratio.name = name
    if train is not None and test is not None:
        try:
            train = int(train)
            test = int(test)
        except ValueError:
            return jsonify({'error': 'train dan test harus angka'}), 400
        if train + test != 100:
            return jsonify({'error': 'Total harus 100'}), 400
        ratio.train_pct = train
        ratio.test_pct = test
    db.session.commit()
    return jsonify(ratio.to_dict()), 200

# DELETE
@split_ratio_bp.route('/<int:ratio_id>', methods=['DELETE'])
def delete(ratio_id):
    ratio = SplitRatio.query.get(ratio_id)
    if not ratio:
        return jsonify({'error': 'Split ratio not found'}), 404
    db.session.delete(ratio)
    db.session.commit()
    return jsonify({'message': 'Split ratio deleted'}), 200