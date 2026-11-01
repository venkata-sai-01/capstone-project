import importlib
import os
import sqlite3
import threading
import time
from typing import Any, Set
from urllib.parse import unquote, urlparse

import uvicorn
from fastapi import FastAPI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
VIDEO_UPLOAD_DIR = os.path.join(UPLOADS_DIR, "videos")
CITIZEN_UPLOAD_DIR = os.path.join(UPLOADS_DIR, "citizen_uploads")
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "evidence", "violations")
CSV_LOG_FILE = os.path.join(DATA_DIR, "traffic_violations_log.csv")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "assets", "models", "yolo26n.pt")


def _project_path(path: str) -> str:
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _load_local_environment() -> None:
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(env_path):
        return

    with open(env_path, encoding="utf-8") as env_file:
        for line_number, line in enumerate(env_file, start=1):
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            key, separator, value = entry.partition("=")
            key = key.strip()
            value = value.strip()
            if not separator or not key or not value:
                raise ValueError(f"Invalid .env entry on line {line_number}.")
            if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
                value = value[1:-1]
            os.environ.setdefault(key, value)


_load_local_environment()
MODEL_PATH = _project_path(os.environ.get("YOLO_MODEL_PATH", "assets/models/yolo26n.pt"))
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if "REPLACE_WITH_APP_PASSWORD" in DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL still contains a sample password. Run "
        "'python scripts/configure_postgres.py' from a terminal, or replace the "
        "sample connection string in .env with valid PostgreSQL credentials."
    )
SQLITE_DATABASE_DIR = _project_path(os.environ.get("SQLITE_DATABASE_DIR", "data").strip() or "data")
DASHBOARD_API_URL = os.environ.get("DASHBOARD_API_URL", "").strip().rstrip("/")
DASHBOARD_UI_URL = os.environ.get("DASHBOARD_UI_URL", "http://127.0.0.1:8080").strip().rstrip("/")
DASHBOARD_API_INTERFACE_URL = os.environ.get(
    "DASHBOARD_API_INTERFACE_URL", "http://127.0.0.1:8081"
).strip().rstrip("/")
DASHBOARD_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "DASHBOARD_ORIGINS",
        f"{DASHBOARD_UI_URL},{DASHBOARD_API_INTERFACE_URL},http://localhost:8080,http://localhost:8081",
    ).split(",")
    if origin.strip()
]
_SQLITE_WAL_LOCK = threading.Lock()
_SQLITE_WAL_DATABASES: Set[str] = set()


class _DatabaseCursor:
    def __init__(self, cursor: Any, is_postgres: bool):
        self._cursor = cursor
        self._is_postgres = is_postgres

    def execute(self, query: str, parameters: tuple = ()):
        if self._is_postgres:
            query = query.replace("?", "%s")
            query = query.replace(" LIKE ", " ILIKE ")
        self._cursor.execute(query, parameters)
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()


class _DatabaseConnection:
    def __init__(self, connection: Any, is_postgres: bool):
        self._connection = connection
        self.is_postgres = is_postgres

    def cursor(self) -> _DatabaseCursor:
        return _DatabaseCursor(self._connection.cursor(), self.is_postgres)

    def commit(self):
        self._connection.commit()

    def close(self):
        self._connection.close()


def _connect_database(db_file: str) -> _DatabaseConnection:
    if DATABASE_URL:
        scheme = DATABASE_URL.partition("://")[0].lower()
        if scheme not in {"postgres", "postgresql"}:
            raise ValueError(
                "DATABASE_URL must use the postgres:// or postgresql:// scheme. "
                "For SQLite, leave DATABASE_URL unset and optionally set SQLITE_DATABASE_DIR."
            )
        try:
            postgres_driver = importlib.import_module("pg8000.dbapi")
        except ModuleNotFoundError as exc:
            if exc.name != "pg8000":
                raise
            raise RuntimeError(
                "DATABASE_URL is set, but pg8000 is not installed. Install project "
                "dependencies with 'pip install -r requirements.txt'."
            ) from exc
        parsed_url = urlparse(DATABASE_URL)
        connection = postgres_driver.connect(
            host=parsed_url.hostname or "localhost",
            port=parsed_url.port or 5432,
            database=unquote(parsed_url.path.lstrip("/")),
            user=unquote(parsed_url.username or ""),
            password=unquote(parsed_url.password or ""),
            timeout=10,
        )
        return _DatabaseConnection(connection, is_postgres=True)

    sqlite_path = os.path.abspath(os.path.join(SQLITE_DATABASE_DIR or ".", db_file))
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    connection = sqlite3.connect(sqlite_path, check_same_thread=False, timeout=10.0)
    database_key = sqlite_path
    with _SQLITE_WAL_LOCK:
        if database_key not in _SQLITE_WAL_DATABASES:
            connection.execute("PRAGMA journal_mode=WAL")
            _SQLITE_WAL_DATABASES.add(database_key)
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=10000")
    return _DatabaseConnection(connection, is_postgres=False)


def _database_backend() -> str:
    return "postgresql" if DATABASE_URL else "sqlite"


def _start_auxiliary_server(application: FastAPI, name: str, host: str, port: int):
    server = uvicorn.Server(uvicorn.Config(
        application,
        host=host,
        port=port,
        log_level="info",
    ))
    thread = threading.Thread(target=server.run, name=f"{name}-web-server", daemon=True)
    thread.start()
    startup_deadline = time.monotonic() + 10.0
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError(f"{name} web server failed to start; check its port and logs.")
        if time.monotonic() >= startup_deadline:
            raise TimeoutError(f"{name} web server did not become ready within 10 seconds.")
        time.sleep(0.05)
    return server, thread
