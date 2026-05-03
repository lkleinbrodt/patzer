import os
from pathlib import Path
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from .db import init_db
from .routes import eval_routes, lichess_routes


def create_app() -> Flask:
    # Load .env from the patzer repo root (one level up from dashboard/)
    repo_root = Path(__file__).parent.parent.parent
    load_dotenv(repo_root / ".env")

    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

    init_db()

    app.register_blueprint(eval_routes.bp)
    app.register_blueprint(lichess_routes.bp)

    return app
