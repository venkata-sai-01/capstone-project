import os
from pathlib import Path


def load_local_environment(project_root: str | Path) -> None:
    env_path = Path(project_root) / ".env"
    if not env_path.is_file():
        return

    with env_path.open(encoding="utf-8") as env_file:
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
