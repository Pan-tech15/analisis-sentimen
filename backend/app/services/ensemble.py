# app/services/ensemble.py
import json
import os
import joblib
from app.services.testing_service import predict_indobert_proba, predict_lexicon_proba

class EnsembleService:
    def __init__(self, training_a, training_b, weights_path=None):
        self.training_a = training_a
        self.training_b = training_b

        if weights_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            weights_path = os.path.join(base_dir, 'app', 'data', 'per_class_weights.json')

        with open(weights_path, 'r') as f:
            self.weights = json.load(f)

        self.artifacts_a = joblib.load(training_a.model_path)
        self.artifacts_b = joblib.load(training_b.model_path)
        self.classes = list(self.weights['IndoBERT-KNN'].keys())

    def predict(self, text):
        _, proba_a = predict_indobert_proba([text], self.artifacts_a)
        _, proba_b = predict_lexicon_proba([text], self.artifacts_b)

        proba_a = proba_a[0]
        proba_b = proba_b[0]

        scores = {}
        for i, cls in enumerate(self.classes):
            w_a = self.weights['IndoBERT-KNN'].get(cls, 1.0)
            w_b = self.weights['Lexicon-NB'].get(cls, 1.0)
            scores[cls] = (proba_a[i] * w_a) + (proba_b[i] * w_b)

        predicted_class = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[predicted_class] / total if total > 0 else 0.0
        return predicted_class, confidence, scores