import argparse
import getpass
import secrets
from pathlib import Path
from urllib.parse import quote

import pg8000.dbapi


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_DIR / ".env"


def build_database_url(
    host: str, port: int, database: str, username: str, password: str
) -> str:
    authority_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return (
        f"postgresql://{quote(username, safe='')}:{quote(password, safe='')}"
        f"@{authority_host}:{port}/{quote(database, safe='')}"
    )


def save_database_url(database_url: str) -> None:
    existing = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    output = []
    replaced = False
    for line in existing:
        if line.partition("=")[0].strip() == "DATABASE_URL":
            if not replaced:
                output.append(f"DATABASE_URL={database_url}")
                replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"DATABASE_URL={database_url}")
    ENV_FILE.write_text("\n".join(output) + "\n", encoding="utf-8")


def quote_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("PostgreSQL database and role names must be non-empty.")
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    if "\x00" in value:
        raise ValueError("PostgreSQL password contains a null character.")
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dedicated PostgreSQL role and database for the traffic safety app. "
            "The administrator password is requested securely in the terminal."
        )
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--admin-user", default="postgres")
    parser.add_argument("--database", default="traffic_safety")
    parser.add_argument("--app-user", default="traffic_app")
    args = parser.parse_args()

    admin_password = getpass.getpass(
        f"PostgreSQL administrator password for {args.admin_user}@{args.host}: "
    )
    app_password = secrets.token_urlsafe(32)

    try:
        admin_connection = pg8000.dbapi.connect(
            database="postgres",
            host=args.host,
            port=args.port,
            user=args.admin_user,
            password=admin_password,
            timeout=10,
        )
        admin_connection.autocommit = True
        try:
            cursor = admin_connection.cursor()
            try:
                cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (args.app_user,))
                if cursor.fetchone():
                    raise RuntimeError(
                        f"PostgreSQL role {args.app_user!r} already exists. "
                        "Choose another --app-user; this setup will not reset an existing role."
                    )
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s", (args.database,)
                )
                if cursor.fetchone():
                    raise RuntimeError(
                        f"PostgreSQL database {args.database!r} already exists. "
                        "Choose another --database; this setup will not alter existing data."
                    )
                cursor.execute(
                    f"CREATE ROLE {quote_identifier(args.app_user)} LOGIN "
                    f"PASSWORD {quote_literal(app_password)}"
                )
                cursor.execute(
                    f"CREATE DATABASE {quote_identifier(args.database)} "
                    f"OWNER {quote_identifier(args.app_user)}"
                )
            finally:
                cursor.close()
        finally:
            admin_connection.close()

        database_url = build_database_url(
            args.host, args.port, args.database, args.app_user, app_password
        )
        app_connection = pg8000.dbapi.connect(
            host=args.host,
            port=args.port,
            database=args.database,
            user=args.app_user,
            password=app_password,
            timeout=10,
        )
        app_connection.close()
        save_database_url(database_url)
    except (pg8000.dbapi.DatabaseError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"PostgreSQL setup failed: {exc}\n"
            "No existing role or database is overwritten. Check the administrator "
            "credentials and PostgreSQL service, then retry."
        ) from exc

    print(
        f"PostgreSQL is configured for database {args.database!r} using role "
        f"{args.app_user!r}. DATABASE_URL was saved to the ignored local .env file."
    )


if __name__ == "__main__":
    main()
