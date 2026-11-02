import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.services.india_vehicle_codes import (
    LEGACY_STATE_CODES,
    RTO_CODES,
    STATE_CODES,
    STATE_TYPES,
    lookup_indian_vehicle_code,
)


class IndiaVehicleCodeTests(unittest.TestCase):
    def test_includes_all_current_states_and_union_territories(self) -> None:
        self.assertEqual(len(STATE_CODES), 36)
        self.assertEqual(
            sum(state_type == "state" for state_type in STATE_TYPES.values()),
            28,
        )
        self.assertEqual(
            sum(state_type == "union_territory" for state_type in STATE_TYPES.values()),
            8,
        )

    def test_resolves_current_and_legacy_registration_prefixes(self) -> None:
        current = lookup_indian_vehicle_code("TG 09 BK 1234")
        legacy = lookup_indian_vehicle_code("TS09BK1234")

        self.assertIsNotNone(current)
        self.assertEqual(current["state_name"], "Telangana")
        self.assertEqual(current["district_code"], "TG 09")
        self.assertEqual(current["rto_area"], "Khairtabad (Hyderabad)")
        self.assertEqual(legacy["state_name"], "Telangana")
        self.assertEqual(legacy["rto_area"], current["rto_area"])
        self.assertEqual(LEGACY_STATE_CODES["TS"], "TG")

    def test_resolves_rto_area_for_legacy_and_union_territory_codes(self) -> None:
        odisha = lookup_indian_vehicle_code("OD 02 AB 1234")
        ladakh = lookup_indian_vehicle_code("LA01AB1234")

        self.assertEqual(odisha["state_name"], "Odisha")
        self.assertEqual(odisha["rto_area"], "Bhubaneswar")
        self.assertIn("Kargil", ladakh["rto_area"])
        self.assertEqual(ladakh["state_type"], "union_territory")
        self.assertGreater(len(RTO_CODES), 1000)

    def test_returns_state_for_unmapped_rto_and_rejects_unknown_prefix(self) -> None:
        unmapped = lookup_indian_vehicle_code("OD99AB1234")

        self.assertEqual(unmapped["state_name"], "Odisha")
        self.assertIsNone(unmapped["rto_area"])
        self.assertIsNone(lookup_indian_vehicle_code("ZZ01AB1234"))
        self.assertIsNone(lookup_indian_vehicle_code("invalid"))


class VehicleRegistryCodeIntegrationTests(unittest.TestCase):
    def test_registry_search_resolves_non_ap_registration_codes(self) -> None:
        from src.services.integrations import APVehicleRegistryEngine

        with TemporaryDirectory() as temporary_directory:
            registry = APVehicleRegistryEngine(
                str(Path(temporary_directory) / "registry.db")
            )
            try:
                results = registry.search_records("TG09BK1234")
            finally:
                registry.conn.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["state_name"], "Telangana")
        self.assertEqual(results[0]["district_code"], "TG 09")
        self.assertEqual(results[0]["rto_area"], "Khairtabad (Hyderabad)")


if __name__ == "__main__":
    unittest.main()
