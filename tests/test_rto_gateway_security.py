import asyncio
import unittest
from unittest.mock import patch

from src.services.integrations import APRTOIntegrationGateway


class RtoGatewaySecurityTests(unittest.TestCase):
    def test_sandbox_dispatch_is_explicitly_local(self) -> None:
        gateway = APRTOIntegrationGateway(env="SANDBOX")
        self.assertTrue(asyncio.run(gateway._send_to_rto({})))
        self.assertEqual(gateway.dispatch_status, "sandbox_simulation")

    def test_production_without_secure_configuration_fails_closed(self) -> None:
        gateway = APRTOIntegrationGateway(
            api_base="http://example.invalid",
            secret_key="short",
            api_token="",
            env="PRODUCTION",
        )
        with patch("src.services.integrations.aiohttp.ClientSession") as client:
            self.assertFalse(asyncio.run(gateway._send_to_rto({})))
        client.assert_not_called()
        self.assertEqual(
            gateway.dispatch_status,
            "production_dispatch_disabled_missing_secure_configuration",
        )

    def test_production_requires_https_even_with_credentials(self) -> None:
        gateway = APRTOIntegrationGateway(
            api_base="http://example.invalid/api",
            secret_key="s" * 48,
            api_token="configured-token",
            env="PRODUCTION",
        )
        with patch("src.services.integrations.aiohttp.ClientSession") as client:
            self.assertFalse(asyncio.run(gateway._send_to_rto({})))
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
