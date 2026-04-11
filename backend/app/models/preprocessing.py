from app import db
from datetime import datetime

class Preprocessing(db.Model):
    __tablename__ = 'preprocessings'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    preprocessed_filepath = db.Column(db.String(500), nullable=False)
    row_count = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Tambahan untuk progress tracking
    status = db.Column(db.String(50), default='pending')  # pending, running, completed, failed
    progress = db.Column(db.Integer, default=0)           # 0-100

    dataset = db.relationship('Dataset', backref='preprocessings')