import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Tambahkan engine options hanya jika menggunakan PostgreSQL
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith('postgresql'):
        SQLALCHEMY_ENGINE_OPTIONS = {
            'connect_args': {
                'options': '-c timezone=UTC'
            }
        }
    else:
        SQLALCHEMY_ENGINE_OPTIONS = {}
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-dev-secret-key')
    JWT_TOKEN_LOCATION = ['headers']
    JWT_ACCESS_TOKEN_EXPIRES = 7200  # 1 jam