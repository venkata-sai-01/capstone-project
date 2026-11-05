import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import VIDEO_UPLOAD_DIR, app, vahan_db


VIEWER_TOKEN = "v" * 48
OPERATOR_TOKEN = "o" * 48


class ProtectedApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            os.environ,
            {
                "TRAFFIC_VIEWER_TOKEN": VIEWER_TOKEN,
                "TRAFFIC_OPERATOR_TOKEN": OPERATOR_TOKEN,
            },
            clear=False,
        )
        self.environment.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.environment.stop()

    def test_protected_data_requires_a_token(self) -> None:
        response = self.client.get("/api/vahan_lookup", params={"query": "demo"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_missing_server_tokens_fail_closed_and_health_root_stays_public(self) -> None:
        with patch.dict(
            os.environ,
            {"TRAFFIC_VIEWER_TOKEN": "", "TRAFFIC_OPERATOR_TOKEN": ""},
        ):
            response = self.client.get(
                "/api/vahan_lookup",
                params={"query": "demo"},
                headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_viewer_can_read_records_but_cannot_change_sources(self) -> None:
        with patch.object(vahan_db, "search_records", return_value=[]):
            response = self.client.get(
                "/api/vahan_lookup",
                params={"query": "demo"},
                headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
            )
        self.assertEqual(response.status_code, 200)

        response = self.client.post(
            "/api/switch_source",
            params={"source": "sim"},
            headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
        )
        self.assertEqual(response.status_code, 403)

    def test_operator_can_change_sources(self) -> None:
        response = self.client.post(
            "/api/switch_source",
            params={"source": "sim"},
            headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"},
        )
        self.assertEqual(response.status_code, 200)

    def test_upload_rejects_unsupported_extensions_and_oversized_files(self) -> None:
        headers = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
        response = self.client.post(
            "/api/upload_video",
            headers=headers,
            files={"file": ("payload.exe", b"not a video", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 415)

        with patch("src.main.MAX_VIDEO_UPLOAD_BYTES", 3):
            response = self.client.post(
                "/api/upload_video",
                headers=headers,
                files={"file": ("clip.mp4", b"four", "video/mp4")},
            )
        self.assertEqual(response.status_code, 413)

    def test_upload_rejects_undecodable_content_and_cleans_temporary_file(self) -> None:
        headers = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
        before = set(os.listdir(VIDEO_UPLOAD_DIR))
        response = self.client.post(
            "/api/upload_video",
            headers=headers,
            files={"file": ("clip.mp4", b"not a video", "video/mp4")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(set(os.listdir(VIDEO_UPLOAD_DIR)), before)


if __name__ == "__main__":
    unittest.main()
