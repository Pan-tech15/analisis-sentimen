from app import db
from datetime import datetime, timezone

class Training(db.Model):
    __tablename__ = 'trainings'

    id = db.Column(db.Integer, primary_key=True)
    model_config_id = db.Column(db.Integer, db.ForeignKey('model_configs.id', ondelete='CASCADE'), nullable=False)
    dataset_filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(50), default='pending')
    progress = db.Column(db.Integer, default=0)
    metrics = db.Column(db.JSON)
    model_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    dataset = db.relationship('Dataset', backref='trainings')

    # Gunakan back_populates
    config = db.relationship('ModelConfig', back_populates='trainings', passive_deletes=True)

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

        dataset_name = None
        if self.dataset:
            dataset_name = self.dataset.dataset_name

        split_ratio = None
        if self.config and self.config.params:
            split = self.config.params.get('split', {})
            train = split.get('train')
            test = split.get('test')
            if train is not None and test is not None:
                split_ratio = f"{train}:{test}"

        return {
            'id': self.id,
            'model_config_id': self.model_config_id,
            'config_name': self.config.name if self.config else None,
            'algorithm': self.config.algorithm if self.config else None,
            'dataset_filename': self.dataset_filename,
            'dataset_name': dataset_name,
            'dataset_id': self.dataset_id,
            'status': self.status,
            'progress': self.progress,
            'metrics': self.metrics,
            'model_path': self.model_path,
            'created_at': _fmt(self.created_at),
            'completed_at': _fmt(self.completed_at),
            'split_ratio': split_ratio
        }