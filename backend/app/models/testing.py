from app import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON

class Testing(db.Model):
    __tablename__ = 'testings'
    
    id = db.Column(db.Integer, primary_key=True)
    training_id = db.Column(db.Integer, db.ForeignKey('trainings.id'), nullable=False)
    test_dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    accuracy = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    confusion_matrix = db.Column(JSON)
    tested_at = db.Column(db.DateTime, default=datetime.utcnow)