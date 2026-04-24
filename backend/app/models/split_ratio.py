from app import db
from datetime import datetime

class SplitRatio(db.Model):
    __tablename__ = 'split_ratios'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    train_pct = db.Column(db.Integer, nullable=False)   # menyimpan persen train
    test_pct = db.Column(db.Integer, nullable=False)    # menyimpan persen test
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'train': self.train_pct,
            'test': self.test_pct,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }