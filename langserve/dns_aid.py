"""DNS-AID integration for LangServe.

Provides automatic DNS-based agent discovery publishing for LangServe
applications. When enabled, agents served via ``add_routes`` are
automatically published to DNS using the DNS-AID protocol (SVCB + TXT
records) and unpublished on shutdown.

Setup:
    Install the ``dns-aid`` package:

    .. code-block:: bash

        pip install dns-aid

Usage with add_routes:
    .. code-block:: python

        from fastapi import FastAPI
        from langserve import add_routes
        from langserve.dns_aid import DnsAidConfig, with_dns_aid

        app = FastAPI()
        add_routes(app, my_runnable, path="/my-agent")

        # Wrap app with DNS-AID auto-publish
        app = with_dns_aid(
            app,
            config=DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
                backend_name="route53",
            ),
        )

Usage with lifespan:
    .. code-block:: python

        from contextlib import asynccontextmanager
        from fastapi import FastAPI
        from langserve import add_routes
        from langserve.dns_aid import DnsAidConfig, dns_aid_lifespan

        @asynccontextmanager
        async def lifespan(app):
            async with dns_aid_lifespan(app, config=dns_aid_config):
                yield

        app = FastAPI(lifespan=lifespan)
        add_routes(app, my_runnable, path="/my-agent")
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Sequence, Union

logger = logging.getLogger(__name__)


def _import_dns_aid() -> Any:
    """Lazy import dns_aid, raising a clear error if not installed."""
    try:
        import dns_aid

        return dns_aid
    except ImportError:
        raise ImportError(
            "The dns-aid package is required for DNS-AID integration. "
            "Install it with: pip install dns-aid"
        )


def _import_create_backend() -> Any:
    """Lazy import the dns_aid backend factory."""
    try:
        from dns_aid.backends import create_backend

        return create_backend
    except ImportError:
        raise ImportError(
            "The dns-aid package is required for DNS-AID integration. "
            "Install it with: pip install dns-aid"
        )


@dataclass
class DnsAidRouteConfig:
    """Per-route DNS-AID publishing configuration.

    Override the default agent name or add route-specific capabilities.
    """

    name: Optional[str] = None
    """Agent name override. Defaults to the route path slug."""

    protocol: str = "https"
    """Protocol: 'a2a', 'mcp', or 'https'."""

    capabilities: Optional[list[str]] = None
    """Agent capabilities advertised in DNS."""

    description: Optional[str] = None
    """Human-readable description of this agent."""

    version: str = "1.0.0"
    """Agent version string."""


@dataclass
class DnsAidConfig:
    """Configuration for DNS-AID auto-publishing in LangServe.

    Attributes:
        domain: DNS domain to publish agents under
            (e.g. 'agents.example.com').
        endpoint: Hostname where the LangServe app is reachable
            (e.g. 'api.example.com').
        port: Port number the service listens on.
        backend_name: DNS backend name
            (e.g. 'route53', 'cloudflare', 'ddns').
        backend: Pre-configured DNSBackend instance.
            Takes priority over backend_name.
        default_protocol: Default protocol for published agents.
        default_capabilities: Default capabilities for all routes.
        default_version: Default version string for all routes.
        ttl: DNS record TTL in seconds.
        route_configs: Per-route overrides keyed by route path.
        auto_publish: Whether to publish on startup.
            Set to False to use manual publishing.
    """

    domain: str = ""
    endpoint: str = ""
    port: int = 443
    backend_name: Optional[str] = None
    backend: Any = None
    default_protocol: str = "https"
    default_capabilities: Optional[list[str]] = None
    default_version: str = "1.0.0"
    ttl: int = 3600
    route_configs: dict[str, DnsAidRouteConfig] = field(
        default_factory=dict
    )
    auto_publish: bool = True


def _path_to_agent_name(path: str) -> str:
    """Convert a LangServe route path to a DNS-safe agent name.

    Examples:
        '/my-agent' -> 'my-agent'
        '/api/v1/chat' -> 'api-v1-chat'
        '' -> 'default'
    """
    name = path.strip("/")
    if not name:
        return "default"
    return name.replace("/", "-").replace("_", "-").lower()


def _get_backend(config: DnsAidConfig) -> Any:
    """Resolve a DNS backend from the config."""
    if config.backend is not None:
        return config.backend
    if config.backend_name:
        create_backend = _import_create_backend()
        return create_backend(config.backend_name)
    return None


async def publish_routes(
    paths: Sequence[str],
    config: DnsAidConfig,
) -> list[dict[str, Any]]:
    """Publish LangServe routes as DNS-AID agent records.

    Args:
        paths: Route paths registered with add_routes
            (e.g. ['/my-agent', '/chat']).
        config: DNS-AID configuration.

    Returns:
        List of publish results (one per route).
    """
    dns_aid = _import_dns_aid()
    backend = _get_backend(config)
    results = []

    for path in paths:
        route_config = config.route_configs.get(path, DnsAidRouteConfig())
        agent_name = route_config.name or _path_to_agent_name(path)
        protocol = route_config.protocol or config.default_protocol
        capabilities = (
            route_config.capabilities or config.default_capabilities
        )
        description = route_config.description
        version = route_config.version or config.default_version

        try:
            result = await dns_aid.publish(
                name=agent_name,
                domain=config.domain,
                protocol=protocol,
                endpoint=config.endpoint,
                port=config.port,
                capabilities=capabilities,
                version=version,
                description=description,
                ttl=config.ttl,
                backend=backend,
            )
            results.append(result.model_dump())
            logger.info(
                "DNS-AID: Published agent '%s' at %s",
                agent_name,
                config.domain,
            )
        except Exception:
            logger.exception(
                "DNS-AID: Failed to publish agent '%s'", agent_name
            )

    return results


async def unpublish_routes(
    paths: Sequence[str],
    config: DnsAidConfig,
) -> None:
    """Remove DNS-AID records for LangServe routes.

    Args:
        paths: Route paths to unpublish.
        config: DNS-AID configuration.
    """
    dns_aid = _import_dns_aid()
    backend = _get_backend(config)

    for path in paths:
        route_config = config.route_configs.get(path, DnsAidRouteConfig())
        agent_name = route_config.name or _path_to_agent_name(path)
        protocol = route_config.protocol or config.default_protocol

        try:
            await dns_aid.unpublish(
                name=agent_name,
                domain=config.domain,
                protocol=protocol,
                backend=backend,
            )
            logger.info(
                "DNS-AID: Unpublished agent '%s' from %s",
                agent_name,
                config.domain,
            )
        except Exception:
            logger.exception(
                "DNS-AID: Failed to unpublish agent '%s'", agent_name
            )


@asynccontextmanager
async def dns_aid_lifespan(
    app: Any,
    config: DnsAidConfig,
) -> AsyncIterator[None]:
    """Async context manager for DNS-AID lifecycle management.

    Publishes all registered LangServe routes on startup and
    unpublishes them on shutdown.

    Use this in your FastAPI lifespan:

    .. code-block:: python

        @asynccontextmanager
        async def lifespan(app):
            async with dns_aid_lifespan(app, config=my_config):
                yield

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.
        config: DNS-AID configuration.
    """
    from langserve.server import _APP_TO_PATHS

    paths = list(_APP_TO_PATHS.get(app, set()))

    if config.auto_publish and paths:
        await publish_routes(paths, config)

    try:
        yield
    finally:
        if config.auto_publish and paths:
            await unpublish_routes(paths, config)


def with_dns_aid(
    app: Any,
    config: DnsAidConfig,
) -> Any:
    """Wrap a FastAPI app with DNS-AID auto-publish lifecycle.

    This replaces the app's lifespan with one that publishes
    routes on startup and unpublishes on shutdown.

    Args:
        app: FastAPI application with routes already added.
        config: DNS-AID configuration.

    Returns:
        The same app instance with DNS-AID lifespan attached.

    Example:
        .. code-block:: python

            app = FastAPI()
            add_routes(app, my_runnable, path="/agent")
            app = with_dns_aid(app, config=DnsAidConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
                backend_name="route53",
            ))
    """
    try:
        from fastapi import FastAPI as FastAPIClass
    except ImportError:
        raise ImportError(
            "FastAPI is required for with_dns_aid. "
            "Install it with: pip install 'langserve[server]'"
        )

    if not isinstance(app, FastAPIClass):
        raise TypeError(
            "with_dns_aid requires a FastAPI app instance, "
            f"got {type(app).__name__}. For APIRouter, use "
            "dns_aid_lifespan directly."
        )

    existing_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(app_: Any) -> AsyncIterator[None]:
        async with dns_aid_lifespan(app_, config=config):
            if existing_lifespan is not None:
                async with existing_lifespan(app_) as maybe_state:
                    yield maybe_state
            else:
                yield

    app.router.lifespan_context = combined_lifespan
    return app
