import os
import json
import logging
from app import db
from app.models.testing import Testing
from app.models.training import Training
from app.models.model_config import ModelConfig
from app.utils.metrics_utils import f1_score_from_confusion_matrix

logger = logging.getLogger(__name__)

def update_ensemble_weights():
    """Menghitung ulang bobot ensemble berdasarkan testing dengan akurasi tertinggi dari kedua model"""
    weights = {}
    
    for algorithm in ['IndoBERT-KNN', 'Lexicon-NB']:
        # Ambil testing dengan akurasi tertinggi yang completed untuk algorithm ini
        test = Testing.query.join(Training).join(ModelConfig).filter(
            ModelConfig.algorithm == algorithm,
            Testing.status == 'completed',
            Testing.accuracy.isnot(None)
        ).order_by(Testing.accuracy.desc()).first()
        
        if not test:
            logger.warning(f"Tidak ada testing completed dengan accuracy untuk {algorithm}, lewati.")
            continue
        
        # Ambil confusion matrix dan class labels
        cm = test.confusion_matrix
        if not cm:
            logger.warning(f"Tidak ada confusion_matrix untuk testing {test.id} ({algorithm})")
            continue
        
        # Prioritas: class_labels dari metrics, lalu dari training metrics
        class_labels = test.metrics.get('class_labels') if test.metrics else None
        if not class_labels:
            if test.training and test.training.metrics:
                class_labels = test.training.metrics.get('class_labels')
        if not class_labels:
            logger.warning(f"Tidak ada class_labels untuk {algorithm}")
            continue
        
        f1_list = f1_score_from_confusion_matrix(cm)
        if len(f1_list) != len(class_labels):
            logger.error(f"Panjang mismatch: f1_list={len(f1_list)}, class_labels={len(class_labels)}")
            continue
        
        weights[algorithm] = {cls: round(f1, 4) for cls, f1 in zip(class_labels, f1_list)}
        logger.info(f"Bobot {algorithm} dihitung dari testing ID {test.id} (akurasi={test.accuracy:.4f}): {weights[algorithm]}")
    
    if not weights:
        logger.error("Tidak ada bobot yang bisa dihitung, file tidak akan diupdate.")
        return False
    
    # Simpan ke file JSON
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_path = os.path.join(base_dir, 'app', 'data', 'per_class_weights.json')
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    logger.info(f"Bobot ensemble berhasil disimpan ke {weights_path}")
    return True