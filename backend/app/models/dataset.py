from app import db
from datetime import datetime

class Dataset(db.Model):
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=True)  
    dataset_name = db.Column(db.String(200), nullable=True)
    filepath = db.Column(db.String(500), nullable=False)
    row_count = db.Column(db.Integer, default=0)
    has_idiom_count = db.Column(db.Integer, default=0)
    no_idiom_count = db.Column(db.Integer, default=0)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    preprocessed = db.relationship('Preprocessing', backref='original_dataset', lazy=True)