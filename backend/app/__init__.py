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

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    bcrypt.init_app(app)

    CORS(app, 
     resources={r"/api/*": {"origins": ["http://localhost:5500", "http://127.0.0.1:5500"]}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])  # Sesuaikan dengan port frontend nanti

    # Import models
    from app.models import User, Dataset, Idiom, Preprocessing, ModelConfig, Training, Testing

    # Register blueprint
    from app.routes.auth import auth_bp
    app.register_blueprint(auth_bp)

    # Dataset blueprint
    from app.routes.dataset import dataset_bp
    app.register_blueprint(dataset_bp)

    from app.routes.idiom import idiom_bp
    app.register_blueprint(idiom_bp)

    from app.routes.preprocess import preprocess_bp
    app.register_blueprint(preprocess_bp)

    # Route test
    @app.route('/')
    def index():
        return {'message': 'Backend is running'}

    return app