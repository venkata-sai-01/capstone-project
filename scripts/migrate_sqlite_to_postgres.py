import csv
import os
import sqlite3
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import unquote, urlparse

import pg8000.dbapi


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
TABLES = {
    "challan_blockchain.db": (
        "blockchain_blocks",
        """
        CREATE TABLE IF NOT EXISTS blockchain_blocks (
            block_index INTEGER PRIMARY KEY,
            timestamp TEXT,
            challan_no TEXT,
            plate_number TEXT,
            payload_json TEXT,
            prev_hash TEXT,
            merkle_root TEXT,
            block_hash TEXT,
            nonce BIGINT
        )
        """,
    ),
    "citizen_safety.db": (
        "citizen_reports",
        """
        CREATE TABLE IF NOT EXISTS citizen_reports (
            report_id TEXT PRIMARY KEY,
            timestamp TEXT,
            report_type TEXT,
            location TEXT,
            description TEXT,
            plate_number TEXT,
            contact_phone TEXT,
            status TEXT,
            media_path TEXT
        )
        """,
    ),
    "traffic_registry.db": (
        "vahan_registry",
        """
        CREATE TABLE IF NOT EXISTS vahan_registry (
            plate TEXT PRIMARY KEY,
            owner_name TEXT,
            dl_number TEXT,
            district TEXT,
            dto_office TEXT,
            region TEXT,
            pin_code INTEGER,
            make_model TEXT,
            veh_class TEXT,
            fuel_type TEXT,
            fastag_id TEXT,
            insurance_status TEXT,
            puc_status TEXT,
            reg_date TEXT
        )
        """,
    ),
    "traffic_timeseries.db": (
        "highway_telemetry",
        """
        CREATE TABLE IF NOT EXISTS highway_telemetry (
            timestamp_ms BIGINT,
            track_id INTEGER,
            plate TEXT,
            speed_kmh DOUBLE PRECISION,
            lane_id INTEGER,
            status TEXT
        )
        """,
    ),
}


class PostgresOptions(TypedDict):
    host: str
    port: int
    database: str
    user: str
    password: str
    timeout: int


VIOLATION_COLUMNS = (
    "timestamp",
    "challan_no",
    "license_plate",
    "owner_name",
    "driving_license",
    "vehicle_class",
    "district",
    "dto_office",
    "make_model",
    "fuel_type",
    "fastag_id",
    "recorded_speed_kmh",
    "speed_limit_kmh",
    "statutory_section",
    "telugu_notice",
    "fine_amount_inr",
    "cctns_status",
    "digital_signature",
    "snapshot_path",
)
CSV_TO_COLUMN = dict(
    zip(
        (
            "Timestamp",
            "Challan_No",
            "License_Plate",
            "Owner_Name",
            "Driving_License",
            "Vehicle_Class",
            "District",
            "DTO_Office",
            "Make_Model",
            "Fuel_Type",
            "FASTag_ID",
            "Recorded_Speed_KMH",
            "Speed_Limit_KMH",
            "Statutory_Section",
            "Telugu_Notice",
            "Fine_Amount_INR",
            "CCTNS_Status",
            "Digital_Signature",
            "Snapshot_Path",
        ),
        VIOLATION_COLUMNS,
    )
)


def load_database_url() -> str:
    env_file = PROJECT_DIR / ".env"
    if env_file.is_file():
        for line_number, line in enumerate(
            env_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
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


def postgres_connection_options(database_url: str) -> PostgresOptions:
    parsed_url = urlparse(database_url)
    return {
        "host": parsed_url.hostname or "localhost",
        "port": parsed_url.port or 5432,
        "database": unquote(parsed_url.path.lstrip("/")),
        "user": unquote(parsed_url.username or ""),
        "password": unquote(parsed_url.password or ""),
        "timeout": 10,
    }


def create_destination_tables(cursor: Any) -> None:
    for _, (table, create_sql) in TABLES.items():
        cursor.execute(create_sql)
        if table == "highway_telemetry":
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_telemetry_unique_point
                ON highway_telemetry(
                    timestamp_ms, track_id, plate, speed_kmh, lane_id, status
                )
                """
            )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS traffic_violations (
            timestamp TEXT,
            challan_no TEXT PRIMARY KEY,
            license_plate TEXT,
            owner_name TEXT,
            driving_license TEXT,
            vehicle_class TEXT,
            district TEXT,
            dto_office TEXT,
            make_model TEXT,
            fuel_type TEXT,
            fastag_id TEXT,
            recorded_speed_kmh DOUBLE PRECISION,
            speed_limit_kmh INTEGER,
            statutory_section TEXT,
            telugu_notice TEXT,
            fine_amount_inr INTEGER,
            cctns_status TEXT,
            digital_signature TEXT,
            snapshot_path TEXT
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_plate ON vahan_registry(plate)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_telemetry_track_timestamp "
        "ON highway_telemetry(track_id, timestamp_ms)"
    )


def migrate_sqlite_table(
    cursor: Any,
    database_file: Path,
    table: str,
) -> tuple[int, int]:
    if not database_file.is_file():
        raise FileNotFoundError(f"SQLite source database not found: {database_file}")
    with sqlite3.connect(database_file) as source:
        source.row_factory = sqlite3.Row
        source_count = source.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        columns = [
            row["name"]
            for row in source.execute(f'PRAGMA table_info("{table}")').fetchall()
        ]
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join("%s" for _ in columns)
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        before_count = cursor.fetchone()[0]
        insert_sql = (
            f'INSERT INTO "{table}" ({quoted_columns}) VALUES ({placeholders}) '
            "ON CONFLICT DO NOTHING"
        )
        read_cursor = source.execute(f'SELECT {quoted_columns} FROM "{table}"')
        while batch := read_cursor.fetchmany(500):
            cursor.executemany(insert_sql, [tuple(row) for row in batch])
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        migrated_count = cursor.fetchone()[0] - before_count
    return source_count, migrated_count


def numeric_value(value: str | None, converter: Any) -> Any:
    if not value:
        return None
    try:
        return converter(value)
    except ValueError:
        return None


def migrate_violations(cursor: Any) -> tuple[int, int]:
    csv_file = DATA_DIR / "traffic_violations_log.csv"
    if not csv_file.is_file():
        raise FileNotFoundError(f"Violation CSV source not found: {csv_file}")
    cursor.execute("SELECT COUNT(*) FROM traffic_violations")
    before_count = cursor.fetchone()[0]
    placeholders = ", ".join("%s" for _ in VIOLATION_COLUMNS)
    columns = ", ".join(VIOLATION_COLUMNS)
    insert_sql = (
        f"INSERT INTO traffic_violations ({columns}) VALUES ({placeholders}) "
        "ON CONFLICT (challan_no) DO NOTHING"
    )
    source_count = 0
    with csv_file.open("r", newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            source_count += 1
            values = [row.get(header, "") or None for header in CSV_TO_COLUMN]
            values[11] = numeric_value(values[11], float)
            values[12] = numeric_value(values[12], int)
            values[15] = numeric_value(values[15], int)
            cursor.execute(insert_sql, values)
    cursor.execute("SELECT COUNT(*) FROM traffic_violations")
    migrated_count = cursor.fetchone()[0] - before_count
    return source_count, migrated_count


def main() -> None:
    database_url = load_database_url()
    if not database_url:
        raise SystemExit(
            "DATABASE_URL is not set. Run scripts/configure_postgres.py first."
        )
    if not database_url.startswith(("postgres://", "postgresql://")):
        raise SystemExit("DATABASE_URL must point to PostgreSQL.")

    connection = pg8000.dbapi.connect(
        **postgres_connection_options(database_url)
    )
    try:
        cursor = connection.cursor()
        try:
            create_destination_tables(cursor)
            results = []
            for database_file, (table, _) in TABLES.items():
                source_count, migrated_count = migrate_sqlite_table(
                    cursor, DATA_DIR / database_file, table
                )
                results.append((table, source_count, migrated_count))
            source_count, migrated_count = migrate_violations(cursor)
            results.append(("traffic_violations", source_count, migrated_count))
        finally:
            cursor.close()
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print("SQLite-to-PostgreSQL migration complete (existing PostgreSQL rows preserved).")
    for table, source_count, migrated_count in results:
        print(
            f"{table}: {migrated_count} inserted; "
            f"{source_count - migrated_count} already present or skipped"
        )


if __name__ == "__main__":
    main()
