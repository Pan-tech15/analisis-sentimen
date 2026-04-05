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

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    bcrypt.init_app(app)
    CORS(app, origins="http://localhost:5500")  # Sesuaikan dengan port frontend nanti

    # Import models
    from app.models import User, Dataset, Idiom, Preprocessing, ModelConfig, Training, Testing

    # Register blueprint
    from app.routes.auth import auth_bp
    app.register_blueprint(auth_bp)

    # Route test
    @app.route('/')
    def index():
        return {'message': 'Backend is running'}

    return app