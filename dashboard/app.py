import csv
import io
import math
import os
import sqlite3
from collections import deque
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pg8000.dbapi
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
TABLES = {
    "challan_blockchain.db": "blockchain_blocks",
    "citizen_safety.db": "citizen_reports",
    "traffic_registry.db": "vahan_registry",
    "traffic_timeseries.db": "highway_telemetry",
}
CSV_FILE = DATA_DIR / "traffic_violations_log.csv"
MAX_ROWS = 1000


def load_database_url() -> str:
    env_path = PROJECT_DIR / ".env"
    if env_path.is_file():
        with env_path.open(encoding="utf-8") as env_file:
            for line_number, line in enumerate(env_file, start=1):
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                key, separator, value = entry.partition("=")
                if not separator or not key.strip() or not value.strip():
                    raise ValueError(f"Invalid .env entry on line {line_number}.")
                value = value.strip()
                if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
    return os.environ.get("DATABASE_URL", "").strip()


DATABASE_URL = load_database_url()


def postgres_connection_options() -> dict[str, Any]:
    parsed_url = urlparse(DATABASE_URL)
    return {
        "host": parsed_url.hostname or "localhost",
        "port": parsed_url.port or 5432,
        "database": unquote(parsed_url.path.lstrip("/")),
        "user": unquote(parsed_url.username or ""),
        "password": unquote(parsed_url.password or ""),
        "timeout": 10,
    }


@st.cache_data(ttl=30)
def read_table(database: str, table: str) -> tuple[list[str], list[dict], int]:
    if DATABASE_URL:
        if TABLES.get(database) != table:
            raise ValueError(f"Unsupported database table: {database}/{table}")
        try:
            with closing(pg8000.dbapi.connect(**postgres_connection_options())) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        f'SELECT * FROM "{table}" LIMIT %s', (MAX_ROWS,)
                    )
                    raw_rows = cursor.fetchall()
                    columns = (
                        [description[0] for description in cursor.description]
                        if cursor.description
                        else []
                    )
                    cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                    count_row = cursor.fetchone()
                    total = int(count_row[0]) if count_row else 0
                finally:
                    cursor.close()
        except pg8000.dbapi.DatabaseError as exc:
            raise RuntimeError(
                f"Could not read {table} from PostgreSQL: {exc}"
            ) from exc
        return columns, [dict(zip(columns, row)) for row in raw_rows], total

    path = DATA_DIR / database
    if not path.is_file():
        raise FileNotFoundError(f"Database file not found: {path}")

    try:
        with closing(sqlite3.connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            with connection:
                cursor = connection.execute(
                    f'SELECT * FROM "{table}" LIMIT ?', (MAX_ROWS,)
                )
                columns = [description[0] for description in cursor.description or []]
                rows = cursor.fetchall()
                total = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not read {table} from {database}: {exc}") from exc

    return columns, [dict(row) for row in rows], total


@st.cache_data(ttl=30)
def read_violations() -> tuple[list[str], list[dict], int]:
    if DATABASE_URL:
        try:
            with closing(pg8000.dbapi.connect(**postgres_connection_options())) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        "SELECT * FROM traffic_violations "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (MAX_ROWS,),
                    )
                    raw_rows = cursor.fetchall()
                    columns = (
                        [description[0] for description in cursor.description]
                        if cursor.description
                        else []
                    )
                    cursor.execute("SELECT COUNT(*) FROM traffic_violations")
                    count_row = cursor.fetchone()
                    total = int(count_row[0]) if count_row else 0
                finally:
                    cursor.close()
        except pg8000.dbapi.DatabaseError as exc:
            raise RuntimeError(
                f"Could not read traffic violations from PostgreSQL: {exc}"
            ) from exc
        return columns, [dict(zip(columns, row)) for row in raw_rows], total

    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"Violation log not found: {CSV_FILE}")
    try:
        with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            columns = list(reader.fieldnames or [])
            rows = deque(maxlen=MAX_ROWS)
            total = 0
            for row in reader:
                total += 1
                rows.append(row)
            return columns, list(rows), total
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RuntimeError(f"Could not read the violation log: {exc}") from exc


def show_records(
    columns: list[str], rows: list[dict], empty_message: str, alt_text: str
) -> None:
    if not rows:
        st.info(empty_message)
        return
    st.dataframe(
        rows,
        column_order=columns,
        hide_index=True,
        width="stretch",
        alt=alt_text,
    )


def violations_csv(columns: list[str], rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def show_database_table(
    database: str, table: str, empty_message: str, alt_text: str
) -> None:
    try:
        columns, rows, _ = read_table(database, table)
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        return
    show_records(columns, rows, empty_message, alt_text)


def load_counts() -> dict[str, int]:
    counts = {}
    for database, table in TABLES.items():
        try:
            _, _, total = read_table(database, table)
        except (FileNotFoundError, RuntimeError) as exc:
            st.error(str(exc))
            counts[table] = 0
        else:
            counts[table] = total
    try:
        _, _, total = read_violations()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        counts["violations"] = 0
    else:
        counts["violations"] = total
    return counts


def render_overview() -> None:
    st.header("System overview")
    backend = "PostgreSQL" if DATABASE_URL else "local SQLite"
    st.caption(f"A read-only view of traffic and citizen-safety records in {backend}.")
    counts = load_counts()

    first_row = st.columns(3)
    first_row[0].metric("Violation log entries", counts["violations"])
    first_row[1].metric("Registered vehicles", counts["vahan_registry"])
    first_row[2].metric("Citizen reports", counts["citizen_reports"])

    second_row = st.columns(2)
    second_row[0].metric("Highway telemetry records", counts["highway_telemetry"])
    second_row[1].metric("Blockchain blocks", counts["blockchain_blocks"])

    st.subheader("Recent violation log")
    try:
        columns, rows, _ = read_violations()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
    else:
        show_records(
            columns,
            sorted(rows, key=lambda row: row.get("timestamp", ""), reverse=True)[:10],
            "No violation records are available yet.",
            "Recent traffic violation records",
        )


def render_violations() -> None:
    st.header("Traffic violations")
    st.caption(
        f"Showing up to {MAX_ROWS:,} records from "
        f"{'PostgreSQL' if DATABASE_URL else 'the local CSV log'}."
    )
    try:
        columns, rows, _ = read_violations()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        return

    if rows and "Timestamp" in columns:
        rows = sorted(rows, key=lambda row: row.get("Timestamp", ""), reverse=True)
    show_records(columns, rows, "The violation log is empty.", "Traffic violation log")
    if rows:
        st.download_button(
            "Download displayed violation records as CSV",
            data=(
                violations_csv(columns, rows)
                if DATABASE_URL
                else CSV_FILE.read_bytes()
            ),
            file_name="traffic_violations.csv",
            mime="text/csv",
        )


def render_registry() -> None:
    st.header("Vehicle registry")
    search = st.text_input("Filter by license plate")
    try:
        columns, rows, _ = read_table("traffic_registry.db", "vahan_registry")
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        return

    if search:
        term = search.casefold()
        rows = [row for row in rows if term in str(row.get("plate", "")).casefold()]
    show_records(columns, rows, "No vehicles match this filter.", "Vehicle registry records")


def render_citizen_safety() -> None:
    st.header("Citizen safety reports")
    show_database_table(
        "citizen_safety.db",
        "citizen_reports",
        "No citizen reports are available yet.",
        "Citizen safety reports",
    )
    st.caption(
        "Citizen report attachments, if present, are stored in "
        f"{PROJECT_DIR / 'uploads' / 'citizen_uploads'}."
    )


def render_telemetry() -> None:
    st.header("Highway telemetry")
    try:
        columns, rows, _ = read_table("traffic_timeseries.db", "highway_telemetry")
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        return

    if rows:
        speeds = []
        for row in rows:
            try:
                speed = float(row["speed_kmh"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(speed):
                speeds.append(speed)
        metrics = st.columns(3)
        metrics[0].metric("Records shown", len(rows))
        metrics[1].metric(
            "Vehicles tracked", len({row.get("plate") for row in rows if row.get("plate")})
        )
        metrics[2].metric(
            "Average speed", f"{sum(speeds) / len(speeds):.1f} km/h" if speeds else "N/A"
        )
    show_records(
        columns, rows, "No highway telemetry is available yet.", "Highway telemetry records"
    )


def render_blockchain() -> None:
    st.header("Challan blockchain")
    st.caption("Stored block metadata is shown as recorded; this page does not validate hashes.")
    show_database_table(
        "challan_blockchain.db",
        "blockchain_blocks",
        "No blockchain blocks are available yet.",
        "Challan blockchain blocks",
    )


def main() -> None:
    st.set_page_config(page_title="Traffic safety dashboard", layout="wide")
    st.title("Traffic safety dashboard")
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select a section",
        [
            "Overview",
            "Traffic violations",
            "Vehicle registry",
            "Citizen safety",
            "Highway telemetry",
            "Challan blockchain",
        ],
    )

    renderers = {
        "Overview": render_overview,
        "Traffic violations": render_violations,
        "Vehicle registry": render_registry,
        "Citizen safety": render_citizen_safety,
        "Highway telemetry": render_telemetry,
        "Challan blockchain": render_blockchain,
    }
    renderers[page]()
    st.sidebar.caption(
        "Reads from PostgreSQL and local evidence files."
        if DATABASE_URL
        else "Reads local files from the project directory."
    )


if __name__ == "__main__":
    main()
