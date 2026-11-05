import unittest

from src.web.pages import build_api_interface_page, build_dashboard_page


class WebPageApiConfigurationTests(unittest.TestCase):
    def test_dashboard_routes_api_and_websocket_traffic_to_rest_server(self) -> None:
        page = build_dashboard_page(
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8081",
        )

        self.assertIn(
            'window.TRAFFIC_API_BASE = "http://127.0.0.1:8000" || window.location.origin;',
            page,
        )
        self.assertIn(
            'window.TRAFFIC_API_INTERFACE_URL = "http://127.0.0.1:8081";',
            page,
        )

    def test_api_explorer_uses_the_runtime_api_and_dashboard_urls(self) -> None:
        page = build_api_interface_page(
            {"paths": {}},
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8080",
        ).body.decode("utf-8")

        self.assertIn(
            'const configuredApiBase = "http://127.0.0.1:8000";',
            page,
        )
        self.assertIn(
            'const configuredDashboardUrl = "http://127.0.0.1:8080";',
            page,
        )
        self.assertIn("Authorization = `Bearer ${token}`", page)
        self.assertIn("type:'auth', token", page)

    def test_dashboard_uses_session_scoped_tokens_and_marks_local_simulations(self) -> None:
        page = build_dashboard_page(
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8081",
        )

        self.assertIn("sessionStorage.setItem('trafficApiToken'", page)
        self.assertIn("headers.set('Authorization'", page)
        self.assertIn("EXPERIMENTAL LOCAL SIMULATION (NO QUANTUM HARDWARE)", page)
        self.assertIn("no government connection", page)


if __name__ == "__main__":
    unittest.main()
