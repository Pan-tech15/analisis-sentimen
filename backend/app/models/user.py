from app import db
from datetime import datetime

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='admin')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    datasets = db.relationship('Dataset', backref='uploader', lazy=True)