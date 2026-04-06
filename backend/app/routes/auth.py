from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from app.models.user import User
from app import db

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'message': 'Username dan password wajib diisi'}), 400
    
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=str(user.id))
        return jsonify({
            'message': 'Login berhasil',
            'access_token': access_token,
            'username': user.username,
            'role': user.role
        }), 200
    else:
        return jsonify({'message': 'Username atau password salah'}), 401