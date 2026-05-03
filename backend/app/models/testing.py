from app import db
from datetime import datetime, timezone
from sqlalchemy.dialects.postgresql import JSON

class Testing(db.Model):
    __tablename__ = 'testings'

    id = db.Column(db.Integer, primary_key=True)
    training_id = db.Column(db.Integer, db.ForeignKey('trainings.id'), nullable=False)
    test_dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    status = db.Column(db.String(50), default='pending')
    progress = db.Column(db.Integer, default=0)
    accuracy = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    confusion_matrix = db.Column(JSON)
    tested_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))   # ← ubah ke UTC aware

    # Kolom untuk menyimpan semua metrik lengkap
    metrics = db.Column(JSON)

    training = db.relationship('Training', backref='tests', lazy=True)

    def to_dict(self):
        def _fmt(dt):
            """Format a datetime into an ISO 8601 string with 'Z' suffix (UTC)."""
            if dt is None:
                return None
            # Convert to UTC if it has timezone info, otherwise treat naive as UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        result = {
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
            'tested_at': _fmt(self.tested_at),   # ← pakai helper UTC
            'algorithm': self.training.config.algorithm if self.training and self.training.config else None,
            'dataset_filename': self.training.dataset_filename if self.training else None,
            'train_accuracy': self.training.metrics.get('accuracy') if self.training and self.training.metrics else None,
            'model_path': self.training.model_path if self.training else None,
            # Kunci: kirimkan objek metrics utuh
            'metrics': self.metrics
        }

        # (Opsional) Tetap tambahkan field-level untuk backward compatibility
        if self.metrics:
            result.update({
                'macro_accuracy': self.metrics.get('macro_accuracy'),
                'macro_precision': self.metrics.get('macro_precision'),
                'macro_recall': self.metrics.get('macro_recall'),
                'macro_f1_score': self.metrics.get('macro_f1_score'),
                'mcc': self.metrics.get('mcc'),
                'roc_auc': self.metrics.get('roc_auc'),
                'class_labels': self.metrics.get('class_labels')
            })

        return result