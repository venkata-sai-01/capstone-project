import secrets
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_DIR / ".env"
TOKEN_NAMES = ("TRAFFIC_VIEWER_TOKEN", "TRAFFIC_OPERATOR_TOKEN")


def configure_tokens() -> None:
    existing = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    tokens = {name: secrets.token_urlsafe(48) for name in TOKEN_NAMES}
    output = []
    replaced = set()
    for line in existing:
        name, separator, _ = line.partition("=")
        name = name.strip()
        if separator and name in tokens:
            if name not in replaced:
                output.append(f"{name}={tokens[name]}")
                replaced.add(name)
        else:
            output.append(line)
    for name in TOKEN_NAMES:
        if name not in replaced:
            output.append(f"{name}={tokens[name]}")
    ENV_FILE.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(
        "Generated separate viewer and operator access tokens in the ignored local .env file. "
        "Keep them private; the operator token can also perform viewer actions."
    )


if __name__ == "__main__":
    configure_tokens()
