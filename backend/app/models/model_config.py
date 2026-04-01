from app import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON

class ModelConfig(db.Model):
    __tablename__ = 'model_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    algorithm = db.Column(db.String(50), nullable=False)  # 'indobert_knn' or 'naivebayes_lexicon'
    parameters = db.Column(JSON, nullable=False)  # store as JSON
    train_dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)  # preprocessed dataset id
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    trainings = db.relationship('Training', backref='model_config', lazy=True)