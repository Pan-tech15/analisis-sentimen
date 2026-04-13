from app import db
from datetime import datetime
import json

class ModelConfig(db.Model):
    """Tabel untuk menyimpan konfigurasi model yang dibuat pengguna."""
    __tablename__ = 'model_configs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)  # Nama model
    algorithm = db.Column(db.String(50), nullable=False)           # 'IndoBERT-KNN' atau 'Lexicon-NB'
    params = db.Column(db.JSON, nullable=False)                    # Parameter dalam format JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relasi ke Training (satu konfigurasi bisa dipakai berkali-kali)
    trainings = db.relationship('Training', backref='config', lazy=True)

    def __repr__(self):
        return f"<ModelConfig {self.name} ({self.algorithm})>"

    def to_dict(self):
        """Mengembalikan representasi dictionary untuk response API."""
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
        """
        Validasi parameter berdasarkan algoritma yang dipilih.
        Kelompok Lexicon+NaiveBayes dapat menambahkan validasi mereka di sini.
        """
        if algorithm == 'IndoBERT-KNN':
            required_sections = ['general', 'split', 'indobert', 'umap', 'knn']
            for section in required_sections:
                if section not in params:
                    raise ValueError(f"Parameter '{section}' harus ada untuk IndoBERT-KNN.")
        elif algorithm == 'Lexicon-NB':
            required_sections = ['general', 'split', 'naivebayes', 'fusion']
            for section in required_sections:
                if section not in params:
                    raise ValueError(f"Parameter '{section}' harus ada untuk Lexicon-NB.")
        else:
            raise ValueError("Algoritma tidak dikenal.")
        return True