"""Unit tests for DNS-AID LangServe integration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langserve.dns_aid import (
    DnsAidConfig,
    DnsAidRouteConfig,
    _path_to_agent_name,
    publish_routes,
    unpublish_routes,
)


class TestPathToAgentName:
    """Tests for path-to-agent-name conversion."""

    def test_simple_path(self) -> None:
        assert _path_to_agent_name("/my-agent") == "my-agent"

    def test_nested_path(self) -> None:
        assert _path_to_agent_name("/api/v1/chat") == "api-v1-chat"

    def test_empty_path(self) -> None:
        assert _path_to_agent_name("") == "default"

    def test_root_path(self) -> None:
        assert _path_to_agent_name("/") == "default"

    def test_underscores_converted(self) -> None:
        assert _path_to_agent_name("/my_agent") == "my-agent"

    def test_strips_slashes(self) -> None:
        assert _path_to_agent_name("/agent/") == "agent"

    def test_uppercase_lowered(self) -> None:
        assert _path_to_agent_name("/MyAgent") == "myagent"


class TestDnsAidConfig:
    """Tests for DnsAidConfig."""

    def test_defaults(self) -> None:
        config = DnsAidConfig()
        assert config.domain == ""
        assert config.endpoint == ""
        assert config.port == 443
        assert config.backend_name is None
        assert config.backend is None
        assert config.default_protocol == "https"
        assert config.ttl == 3600
        assert config.auto_publish is True

    def test_custom_values(self) -> None:
        config = DnsAidConfig(
            domain="agents.example.com",
            endpoint="api.example.com",
            port=8080,
            backend_name="route53",
            default_protocol="mcp",
            ttl=7200,
        )
        assert config.domain == "agents.example.com"
        assert config.endpoint == "api.example.com"
        assert config.port == 8080
        assert config.backend_name == "route53"
        assert config.default_protocol == "mcp"
        assert config.ttl == 7200

    def test_route_configs(self) -> None:
        config = DnsAidConfig(
            domain="agents.example.com",
            endpoint="api.example.com",
            route_configs={
                "/chat": DnsAidRouteConfig(
                    name="chatbot",
                    capabilities=["conversation"],
                ),
            },
        )
        assert "/chat" in config.route_configs
        assert config.route_configs["/chat"].name == "chatbot"


class TestDnsAidRouteConfig:
    """Tests for DnsAidRouteConfig."""

    def test_defaults(self) -> None:
        rc = DnsAidRouteConfig()
        assert rc.name is None
        assert rc.protocol == "https"
        assert rc.capabilities is None
        assert rc.description is None
        assert rc.version == "1.0.0"


class TestPublishRoutes:
    """Tests for publish_routes."""

    @pytest.mark.asyncio
    async def test_publishes_all_routes(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "success": True,
            "agent": {"name": "my-agent"},
        }

        with patch(
            "langserve.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            config = DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
            )
            results = await publish_routes(
                ["/my-agent", "/chat"], config
            )

        assert len(results) == 2
        assert mock_dns_aid.publish.await_count == 2

    @pytest.mark.asyncio
    async def test_uses_route_config_override(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"success": True}

        with patch(
            "langserve.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            config = DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
                route_configs={
                    "/chat": DnsAidRouteConfig(
                        name="chatbot",
                        protocol="mcp",
                        capabilities=["conversation"],
                        description="A helpful chatbot",
                    ),
                },
            )
            await publish_routes(["/chat"], config)

        call_kwargs = mock_dns_aid.publish.call_args.kwargs
        assert call_kwargs["name"] == "chatbot"
        assert call_kwargs["protocol"] == "mcp"
        assert call_kwargs["capabilities"] == ["conversation"]
        assert call_kwargs["description"] == "A helpful chatbot"

    @pytest.mark.asyncio
    async def test_defaults_to_path_slug_name(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"success": True}

        with patch(
            "langserve.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            config = DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
            )
            await publish_routes(["/my-agent"], config)

        call_kwargs = mock_dns_aid.publish.call_args.kwargs
        assert call_kwargs["name"] == "my-agent"

    @pytest.mark.asyncio
    async def test_handles_publish_error_gracefully(self) -> None:
        with patch(
            "langserve.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(
                side_effect=Exception("DNS error")
            )
            mock_import.return_value = mock_dns_aid

            config = DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
            )
            # Should not raise
            results = await publish_routes(["/agent"], config)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_passes_backend(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"success": True}
        mock_backend = MagicMock()

        with patch(
            "langserve.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            config = DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
                backend=mock_backend,
            )
            await publish_routes(["/agent"], config)

        call_kwargs = mock_dns_aid.publish.call_args.kwargs
        assert call_kwargs["backend"] is mock_backend


class TestUnpublishRoutes:
    """Tests for unpublish_routes."""

    @pytest.mark.asyncio
    async def test_unpublishes_all_routes(self) -> None:
        with patch(
            "langserve.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.unpublish = AsyncMock(return_value=True)
            mock_import.return_value = mock_dns_aid

            config = DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
            )
            await unpublish_routes(["/agent-1", "/agent-2"], config)

        assert mock_dns_aid.unpublish.await_count == 2

    @pytest.mark.asyncio
    async def test_uses_route_config_name(self) -> None:
        with patch(
            "langserve.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.unpublish = AsyncMock(return_value=True)
            mock_import.return_value = mock_dns_aid

            config = DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
                route_configs={
                    "/chat": DnsAidRouteConfig(name="chatbot"),
                },
            )
            await unpublish_routes(["/chat"], config)

        call_kwargs = mock_dns_aid.unpublish.call_args.kwargs
        assert call_kwargs["name"] == "chatbot"

    @pytest.mark.asyncio
    async def test_handles_unpublish_error_gracefully(self) -> None:
        with patch(
            "langserve.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.unpublish = AsyncMock(
                side_effect=Exception("DNS error")
            )
            mock_import.return_value = mock_dns_aid

            config = DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
            )
            # Should not raise
            await unpublish_routes(["/agent"], config)


class TestDnsAidLifespan:
    """Tests for dns_aid_lifespan context manager."""

    @pytest.mark.asyncio
    async def test_publishes_on_enter_unpublishes_on_exit(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"success": True}

        with patch(
            "langserve.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_dns_aid.unpublish = AsyncMock(return_value=True)
            mock_import.return_value = mock_dns_aid

            from langserve.dns_aid import dns_aid_lifespan
            from langserve import server

            mock_app = MagicMock()
            server._APP_TO_PATHS[mock_app] = {"/agent"}

            config = DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
            )

            async with dns_aid_lifespan(mock_app, config=config):
                # Verify publish was called
                mock_dns_aid.publish.assert_awaited_once()

            # Verify unpublish was called on exit
            mock_dns_aid.unpublish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_when_no_paths(self) -> None:
        from langserve.dns_aid import dns_aid_lifespan

        mock_app = MagicMock()
        # Don't register any paths in _APP_TO_PATHS

        config = DnsAidConfig(
            domain="agents.example.com",
            endpoint="api.example.com",
        )

        # dns_aid is never imported because there are no paths
        async with dns_aid_lifespan(mock_app, config=config):
            pass

        # If we got here without error, no publish was attempted

    @pytest.mark.asyncio
    async def test_skips_when_auto_publish_false(self) -> None:
        from langserve.dns_aid import dns_aid_lifespan
        from langserve import server

        mock_app = MagicMock()
        server._APP_TO_PATHS[mock_app] = {"/agent"}

        config = DnsAidConfig(
            domain="agents.example.com",
            endpoint="api.example.com",
            auto_publish=False,
        )

        # dns_aid is never imported because auto_publish is False
        async with dns_aid_lifespan(mock_app, config=config):
            pass

        # If we got here without error, no publish was attempted


class TestWithDnsAid:
    """Tests for with_dns_aid wrapper."""

    def test_attaches_lifespan(self) -> None:
        from langserve.dns_aid import with_dns_aid

        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = FastAPI()
        config = DnsAidConfig(
            domain="agents.example.com",
            endpoint="api.example.com",
        )

        result = with_dns_aid(app, config=config)
        assert result is app
        # Lifespan should be replaced
        assert app.router.lifespan_context is not None

    def test_rejects_non_fastapi(self) -> None:
        from langserve.dns_aid import with_dns_aid

        config = DnsAidConfig(
            domain="agents.example.com",
            endpoint="api.example.com",
        )

        with pytest.raises(TypeError, match="FastAPI app instance"):
            with_dns_aid("not-an-app", config=config)


class TestImportErrors:
    """Tests for lazy import error handling."""

    def test_import_dns_aid_raises_when_missing(self) -> None:
        from langserve.dns_aid import _import_dns_aid

        with patch.dict("sys.modules", {"dns_aid": None}):
            with pytest.raises(ImportError, match="dns-aid"):
                _import_dns_aid()

    def test_import_create_backend_raises_when_missing(self) -> None:
        from langserve.dns_aid import _import_create_backend

        with patch.dict(
            "sys.modules",
            {"dns_aid": None, "dns_aid.backends": None},
        ):
            with pytest.raises(ImportError, match="dns-aid"):
                _import_create_backend()
