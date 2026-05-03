from app import db
from datetime import datetime, timezone

class Preprocessing(db.Model):
    __tablename__ = 'preprocessings'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    preprocessed_filepath = db.Column(db.String(500), nullable=False)
    row_count = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))   # ← ubah ke UTC aware
    metrics = db.Column(db.JSON)
    
    # Tambahan untuk progress tracking
    status = db.Column(db.String(50), default='pending')
    progress = db.Column(db.Integer, default=0)
    
    # Nama yang dapat diedit oleh user
    name = db.Column(db.String(255), nullable=True)

    dataset = db.relationship('Dataset', backref='preprocessings')

    def to_dict(self):
        def _fmt(dt):
            """Format a datetime into an ISO 8601 string with 'Z' suffix (UTC)."""
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'preprocessed_filepath': self.preprocessed_filepath,
            'row_count': self.row_count,
            'timestamp': _fmt(self.timestamp),
            'metrics': self.metrics,
            'status': self.status,
            'progress': self.progress,
            'name': self.name,
        }