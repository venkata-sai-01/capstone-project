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
        self.assertIn("window.setDashboardAccessToken = async ()", page)
        self.assertIn("Connected as operator", page)
        self.assertIn("/api/auth/session", page)
        self.assertIn("operator access is required for this action", page)
        self.assertIn("onkeydown=\"if(event.key==='Enter') setDashboardAccessToken()\"", page)
        self.assertIn("headers.set('Authorization'", page)
        self.assertIn("no government connection", page)

    def test_local_demo_pages_hide_token_prompts_and_enable_operator_controls(self) -> None:
        dashboard = build_dashboard_page(
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8081",
            local_demo_no_auth=True,
        )
        self.assertIn("TRAFFIC_LOCAL_DEMO_NO_AUTH = true", dashboard)
        self.assertIn("Unauthenticated local-only demo mode is active.", dashboard)
        self.assertNotIn('id="dashboardAccessToken"', dashboard)

        explorer = build_api_interface_page(
            {"paths": {}},
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8080",
            local_demo_no_auth=True,
        ).body.decode("utf-8")
        self.assertIn("localDemoNoAuth = true", explorer)
        self.assertNotIn('id="token"', explorer)
        self.assertIn("Unauthenticated local-only demo mode is active.", explorer)
        self.assertIn(
            "const token = localDemoNoAuth ? '' : document.getElementById('token').value.trim();",
            explorer,
        )


if __name__ == "__main__":
    unittest.main()
