import sys
import os

# Resolve `backend` to this directory's package (dashboard/backend), not cwd-relative.
_dash_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_dash_dir, "..")
sys.path.insert(0, _dash_dir)
sys.path.insert(1, os.path.normpath(_repo_root))

from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
