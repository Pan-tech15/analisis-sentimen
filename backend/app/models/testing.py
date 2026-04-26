from app import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON

class Testing(db.Model):
    __tablename__ = 'testings'

    id = db.Column(db.Integer, primary_key=True)
    training_id = db.Column(db.Integer, db.ForeignKey('trainings.id'), nullable=False)
    test_dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)  # boleh null
    status = db.Column(db.String(50), default='pending')          # pending, running, completed, failed
    progress = db.Column(db.Integer, default=0)                    # 0-100 (opsional)
    accuracy = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    confusion_matrix = db.Column(JSON)
    tested_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relasi (opsional)
    training = db.relationship('Training', backref='tests', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'training_id': self.training_id,
            'test_dataset_id': self.test_dataset_id,
            'status': self.status,
            'progress': self.progress,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'confusion_matrix': self.confusion_matrix,
            'tested_at': self.tested_at.isoformat() if self.tested_at else None,
            'algorithm': self.training.config.algorithm if self.training and self.training.config else None,
            'dataset_filename': self.training.dataset_filename if self.training else None,
            'train_accuracy': self.training.metrics.get('accuracy') if self.training and self.training.metrics else None
        }