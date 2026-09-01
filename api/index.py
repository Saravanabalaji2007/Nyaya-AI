import sys
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

app = None

try:
    from app import app as flask_app
    app = flask_app

except Exception:
    import traceback

    def handler(environ, start_response):
        error = traceback.format_exc()
        body = error.encode("utf-8")

        start_response(
            "500 Internal Server Error",
            [
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(body)))
            ]
        )

        return [body]

    app = handler