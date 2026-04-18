from app import db
from datetime import datetime

class Training(db.Model):
    __tablename__ = 'trainings'

    id = db.Column(db.Integer, primary_key=True)
    model_config_id = db.Column(db.Integer, db.ForeignKey('model_configs.id', ondelete='CASCADE'), nullable=False)
    dataset_filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(50), default='pending')
    progress = db.Column(db.Integer, default=0)
    metrics = db.Column(db.JSON)
    model_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    # Gunakan back_populates
    config = db.relationship('ModelConfig', back_populates='trainings', passive_deletes=True)

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