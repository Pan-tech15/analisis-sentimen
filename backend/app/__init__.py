import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_bcrypt import Bcrypt
from .config import Config

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
bcrypt = Bcrypt()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app.config['UPLOAD_FOLDER'] = 'data/raw'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

    if not app.debug:
        # Untuk production, bisa gunakan handler file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Untuk development, tampilkan di console
        logging.basicConfig(
            level=logging.DEBUG if app.debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    bcrypt.init_app(app)

    CORS(app, 
        origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://192.168.1.11:5500"],
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    # Import models
    from app.models import User, Dataset, Idiom, Preprocessing, ModelConfig, Training, Testing

    # Register blueprints
    from app.routes.auth import auth_bp
    app.register_blueprint(auth_bp)

    from app.routes.preprocess import preprocess_bp
    app.register_blueprint(preprocess_bp)

    from app.routes.dataset import dataset_bp
    app.register_blueprint(dataset_bp)

    from app.routes.idiom import idiom_bp
    app.register_blueprint(idiom_bp)

    from app.routes.pengaturan import processing_bp
    app.register_blueprint(processing_bp)

    from app.routes.training import training_bp
    app.register_blueprint(training_bp, url_prefix='/training')

    from app.routes.split_ratio import split_ratio_bp
    app.register_blueprint(split_ratio_bp)

    from app.routes.testing import testing_bp
    app.register_blueprint(testing_bp, url_prefix='/testing')

    from app.routes.dashboard import dashboard_bp
    app.register_blueprint(dashboard_bp)

    @app.route('/')
    def index():
        return {'message': 'Backend is running'}

    return app

# Buat instance app global agar bisa diimpor di file lain
app = create_app()