import os
import unittest
from unittest.mock import patch

from src.auth import (
    AuthenticationConfigurationError,
    OPERATOR,
    VIEWER,
    authenticate_token,
    configured_tokens,
    required_role,
    role_allows,
)


class ApiAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            os.environ,
            {
                "TRAFFIC_VIEWER_TOKEN": "v" * 48,
                "TRAFFIC_OPERATOR_TOKEN": "o" * 48,
            },
            clear=False,
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()

    def test_authenticates_separate_viewer_and_operator_tokens(self) -> None:
        self.assertEqual(authenticate_token("v" * 48), VIEWER)
        self.assertEqual(authenticate_token("o" * 48), OPERATOR)
        self.assertIsNone(authenticate_token("incorrect"))

    def test_viewers_cannot_perform_mutating_operations(self) -> None:
        self.assertEqual(required_role("GET"), VIEWER)
        self.assertEqual(required_role("POST"), OPERATOR)
        self.assertFalse(role_allows(VIEWER, OPERATOR))
        self.assertTrue(role_allows(OPERATOR, OPERATOR))

    def test_missing_short_or_shared_tokens_are_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AuthenticationConfigurationError):
                configured_tokens()
        with patch.dict(
            os.environ,
            {
                "TRAFFIC_VIEWER_TOKEN": "short",
                "TRAFFIC_OPERATOR_TOKEN": "o" * 48,
            },
            clear=True,
        ):
            with self.assertRaises(AuthenticationConfigurationError):
                configured_tokens()
        with patch.dict(
            os.environ,
            {
                "TRAFFIC_VIEWER_TOKEN": "s" * 48,
                "TRAFFIC_OPERATOR_TOKEN": "s" * 48,
            },
            clear=True,
        ):
            with self.assertRaises(AuthenticationConfigurationError):
                configured_tokens()


if __name__ == "__main__":
    unittest.main()
