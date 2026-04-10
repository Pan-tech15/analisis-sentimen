from app import db
from datetime import datetime

class Training(db.Model):
    __tablename__ = 'trainings'

    id = db.Column(db.Integer, primary_key=True)
    model_config_id = db.Column(db.Integer, db.ForeignKey('model_configs.id'), nullable=False)
    dataset_filename = db.Column(db.String(255), nullable=False)  # Nama file dataset yang diupload
    status = db.Column(db.String(50), default='pending')  # pending, running, completed, failed
    progress = db.Column(db.Integer, default=0)            # 0-100
    metrics = db.Column(db.JSON)                           # Akurasi, F1, dll.
    model_path = db.Column(db.String(255))                 # Path file model terlatih
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    # Relasi ke ModelConfig
    config = db.relationship('ModelConfig', backref='trainings')

    def to_dict(self):
        return {
            'id': self.id,
            'model_config_id': self.model_config_id,
            'config_name': self.config.name if self.config else None,
            'algorithm': self.config.algorithm if self.config else None,
            'dataset_filename': self.dataset_filename,
            'status': self.status,
            'progress': self.progress,
            'metrics': self.metrics,
            'model_path': self.model_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }