from app import db
from datetime import datetime

class ModelConfig(db.Model):
    __tablename__ = 'model_configs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    algorithm = db.Column(db.String(50), nullable=False)
    params = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Ganti backref dengan back_populates
    trainings = db.relationship('Training', back_populates='config', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'algorithm': self.algorithm,
            'params': self.params,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @staticmethod
    def validate_params(algorithm, params):
        if algorithm == 'IndoBERT-KNN':
            required = ['general', 'split', 'indobert', 'umap', 'knn']
        elif algorithm == 'Lexicon-NB':
            required = ['general', 'split', 'lexicon', 'naivebayes', 'fusion']
        else:
            raise ValueError("Algoritma tidak dikenal.")
        for sec in required:
            if sec not in params:
                raise ValueError(f"Parameter '{sec}' harus ada.")
        return True