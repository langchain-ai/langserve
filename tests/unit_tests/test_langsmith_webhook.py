"""Unit tests for LangSmith webhook bridge."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langserve.langsmith_webhook import (
    LangSmithWebhookConfig,
    _verify_signature,
    handle_webhook_event,
)


@pytest.fixture
def config():
    """Basic webhook config with mock backend."""
    return LangSmithWebhookConfig(
        domain="agents.example.com",
        endpoint="api.example.com",
        backend_name="mock",
        webhook_secret="test-secret",
    )


class TestVerifySignature:
    """Tests for HMAC signature verification."""

    def test_valid_signature(self):
        payload = b'{"event": "test"}'
        secret = "my-secret"
        digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert _verify_signature(payload, f"sha256={digest}", secret) is True

    def test_invalid_signature(self):
        payload = b'{"event": "test"}'
        assert _verify_signature(payload, "sha256=invalid", "secret") is False

    def test_wrong_secret(self):
        payload = b'{"event": "test"}'
        digest = hmac.new(b"correct", payload, hashlib.sha256).hexdigest()
        assert _verify_signature(payload, f"sha256={digest}", "wrong") is False


class TestHandleWebhookEvent:
    """Tests for event handling logic."""

    @pytest.mark.asyncio
    async def test_publish_on_project_created(self, config):
        event = {
            "event": "project.created",
            "data": {
                "project": {
                    "name": "My Agent",
                    "description": "A helpful agent",
                }
            },
        }

        mock_result = MagicMock(success=True)
        mock_dns_aid = MagicMock()
        mock_dns_aid.publish = AsyncMock(return_value=mock_result)

        with (
            patch("langserve.langsmith_webhook._import_dns_aid", return_value=mock_dns_aid),
            patch("langserve.langsmith_webhook._import_schema_adapter") as mock_adapter,
            patch("langserve.langsmith_webhook._get_backend", return_value=MagicMock()),
        ):
            mock_adapter.return_value = lambda project, **kw: {
                "name": "my-agent",
                "domain": "agents.example.com",
                "protocol": "https",
                "endpoint": "api.example.com",
            }

            result = await handle_webhook_event(event, config)

        assert result["action"] == "published"
        assert result["success"] is True
        mock_dns_aid.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_unpublish_on_project_archived(self, config):
        event = {
            "event": "project.archived",
            "data": {
                "project": {"name": "My Agent"},
            },
        }

        mock_dns_aid = MagicMock()
        mock_dns_aid.unpublish = AsyncMock(return_value=True)

        with (
            patch("langserve.langsmith_webhook._import_dns_aid", return_value=mock_dns_aid),
            patch("langserve.langsmith_webhook._import_schema_adapter"),
            patch("langserve.langsmith_webhook._get_backend", return_value=MagicMock()),
        ):
            result = await handle_webhook_event(event, config)

        assert result["action"] == "unpublished"
        assert result["success"] is True
        mock_dns_aid.unpublish.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignored_event(self, config):
        event = {
            "event": "run.completed",
            "data": {"project": {"name": "agent"}},
        }

        result = await handle_webhook_event(event, config)
        assert result["action"] == "ignored"

    @pytest.mark.asyncio
    async def test_missing_project_name(self, config):
        event = {
            "event": "project.created",
            "data": {"project": {}},
        }

        result = await handle_webhook_event(event, config)
        assert result["action"] == "skipped"

    @pytest.mark.asyncio
    async def test_project_filter(self, config):
        config.project_filter = lambda p: p.get("name") == "allowed"

        event = {
            "event": "project.created",
            "data": {"project": {"name": "not-allowed"}},
        }

        result = await handle_webhook_event(event, config)
        assert result["action"] == "filtered"

    @pytest.mark.asyncio
    async def test_publish_failure_handled(self, config):
        event = {
            "event": "project.created",
            "data": {"project": {"name": "agent"}},
        }

        mock_dns_aid = MagicMock()
        mock_dns_aid.publish = AsyncMock(side_effect=RuntimeError("DNS error"))

        with (
            patch("langserve.langsmith_webhook._import_dns_aid", return_value=mock_dns_aid),
            patch("langserve.langsmith_webhook._import_schema_adapter") as mock_adapter,
            patch("langserve.langsmith_webhook._get_backend", return_value=MagicMock()),
        ):
            mock_adapter.return_value = lambda project, **kw: {
                "name": "agent",
                "domain": "agents.example.com",
                "protocol": "https",
                "endpoint": "api.example.com",
            }

            result = await handle_webhook_event(event, config)

        assert result["action"] == "publish_failed"
        assert result["success"] is False


class TestLangSmithWebhookConfig:
    """Tests for config defaults."""

    def test_defaults(self):
        config = LangSmithWebhookConfig()
        assert config.webhook_path == "/webhooks/langsmith"
        assert config.default_protocol == "https"
        assert config.auto_publish is True
        assert config.auto_unpublish is True
