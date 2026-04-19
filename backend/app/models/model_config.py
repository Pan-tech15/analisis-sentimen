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

    trainings = db.relationship('Training', back_populates='config', lazy=True, cascade='all, delete-orphan', passive_deletes=True)

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
            required = ['general', 'split', 'indobert', 'knn']
            for sec in required:
                if sec not in params:
                    raise ValueError(f"Parameter '{sec}' harus ada untuk IndoBERT-KNN.")
            if 'umap' in params and not isinstance(params['umap'], dict):
                raise ValueError("Parameter 'umap' harus berupa objek.")
            if 'finetune' in params and not isinstance(params['finetune'], dict):
                raise ValueError("Parameter 'finetune' harus berupa objek.")
            if 'hybrid' in params and not isinstance(params['hybrid'], dict):
                raise ValueError("Parameter 'hybrid' harus berupa objek.")
        elif algorithm == 'Lexicon-NB':
            required = ['general', 'split', 'naivebayes', 'fusion']
            for sec in required:
                if sec not in params:
                    raise ValueError(f"Parameter '{sec}' harus ada untuk Lexicon-NB.")
            fusion = params.get('fusion', {})
            if not isinstance(fusion, dict):
                raise ValueError("Parameter 'fusion' harus berupa objek.")
            if 'method' not in fusion:
                raise ValueError("Parameter 'fusion.method' harus ada.")
            if fusion.get('method') == 'weighted' and 'weight' in fusion:
                try:
                    float(fusion['weight'])
                except (ValueError, TypeError):
                    raise ValueError("Parameter 'fusion.weight' harus berupa angka.")
        else:
            raise ValueError("Algoritma tidak dikenal.")
        return True