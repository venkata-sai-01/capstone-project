import os
import tempfile
import unittest
from unittest.mock import patch


_IMPORT_DATABASES = tempfile.TemporaryDirectory(prefix="traffic-safety-import-tests-")
with patch.dict(
    os.environ,
    {
        "DATABASE_URL": "",
        "SQLITE_DATABASE_DIR": _IMPORT_DATABASES.name,
    },
):
    from src import main


def _close_import_databases() -> None:
    for name in ("blockchain_ledger", "citizen_portal", "vahan_db", "timeseries_logger"):
        connection = getattr(getattr(main, name), "conn", None)
        if connection is not None:
            connection.close()
    _IMPORT_DATABASES.cleanup()


unittest.addModuleCleanup(_close_import_databases)


class PlateNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database_directory = tempfile.TemporaryDirectory(prefix="traffic-safety-registry-tests-")
        self.original_database_directory = main.SQLITE_DATABASE_DIR
        main.SQLITE_DATABASE_DIR = self.database_directory.name
        self.registry = main.APVehicleRegistryEngine("registry.db")

    def tearDown(self) -> None:
        self.registry.conn.close()
        main.SQLITE_DATABASE_DIR = self.original_database_directory
        self.database_directory.cleanup()

    def test_hotlist_matches_case_and_format_variants(self) -> None:
        hotlist = main.APCCTNSHotlistDatabase()

        for plate in ("AP 03 BK 4921", "ap03bk4921", "AP-03-BK-4921"):
            with self.subTest(plate=plate):
                self.assertEqual(
                    hotlist.check_hotlist(plate),
                    hotlist.check_hotlist("AP 03 BK 4921"),
                )

    def test_hotlist_does_not_match_an_unlisted_plate(self) -> None:
        self.assertIsNone(main.APCCTNSHotlistDatabase().check_hotlist("AP 03 BK 9999"))

    def test_normalizer_removes_whitespace_and_hyphens(self) -> None:
        self.assertEqual(
            main.APVehicleRegistryEngine._normalize_plate(" ap\t03-bk 4921 "),
            "AP03BK4921",
        )

    def test_generated_profile_is_saved_with_canonical_plate(self) -> None:
        profile = self.registry.resolve_plate_to_profile("ap 03-bk 4921")

        self.assertEqual(profile["plate"], "AP03BK4921")
        stored_plate = self.registry.conn.cursor().execute(
            "SELECT plate FROM vahan_registry"
        ).fetchone()
        self.assertEqual(stored_plate[0], "AP03BK4921")

    def test_registry_finds_existing_formatted_plate_without_duplicate(self) -> None:
        original_profile = self.registry.resolve_plate_to_profile("AP03BK4921")
        cursor = self.registry.conn.cursor()
        cursor.execute(
            "UPDATE vahan_registry SET plate = ? WHERE plate = ?",
            ("AP 03-BK 4921", "AP03BK4921"),
        )
        self.registry.conn.commit()

        resolved_profile = self.registry.resolve_plate_to_profile("ap 03 bk 4921")
        record_count = cursor.execute("SELECT COUNT(*) FROM vahan_registry").fetchone()[0]

        self.assertEqual(resolved_profile["owner_name"], original_profile["owner_name"])
        self.assertEqual(resolved_profile["plate"], "AP 03-BK 4921")
        self.assertEqual(record_count, 1)


class VehicleClassScopeTests(unittest.TestCase):
    def test_supported_vehicles_are_classified_as_vehicles(self) -> None:
        for class_name in ("car", "motorcycle", "bus", "truck", "tractor"):
            with self.subTest(class_name=class_name):
                self.assertEqual(main.APPoliceRadarSafetyEngine._object_kind(class_name), "vehicle")

    def test_removed_classes_are_not_classified_as_vehicles(self) -> None:
        for class_name in ("bicycle", "auto_rickshaw", "e_rickshaw"):
            with self.subTest(class_name=class_name):
                self.assertNotIn(class_name, main.APPoliceRadarSafetyEngine.VEHICLE_CLASSES)
                self.assertEqual(main.APPoliceRadarSafetyEngine._object_kind(class_name), "object")


if __name__ == "__main__":
    unittest.main()
