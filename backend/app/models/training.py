from app import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON

class Training(db.Model):
    __tablename__ = 'trainings'
    
    id = db.Column(db.Integer, primary_key=True)
    model_config_id = db.Column(db.Integer, db.ForeignKey('model_configs.id'), nullable=False)
    accuracy = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    confusion_matrix = db.Column(JSON)  # store as list of lists
    fold_info = db.Column(JSON)  # e.g., {'fold': 3, 'total_folds': 5} if cross-val
    epoch = db.Column(db.Integer)  # for models that use epochs
    best_model_path = db.Column(db.String(500))  # path to saved model file
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    testings = db.relationship('Testing', backref='training', lazy=True)