from app import db
from datetime import datetime

class Idiom(db.Model):
    __tablename__ = 'idioms'
    id = db.Column(db.Integer, primary_key=True)
    idiom_text = db.Column(db.String(200), nullable=False)  # unique, simpan lowercase
    idiom_meaning = db.Column(db.Text, nullable=False)
    source = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)