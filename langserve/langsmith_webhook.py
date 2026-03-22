"""LangSmith webhook bridge for DNS-AID.

FastAPI handler that consumes LangSmith project events and publishes
or unpublishes DNS-AID records accordingly.

Setup:
    .. code-block:: python

        from fastapi import FastAPI
        from langserve.langsmith_webhook import (
            LangSmithWebhookConfig,
            add_langsmith_webhook,
        )

        app = FastAPI()
        add_langsmith_webhook(
            app,
            config=LangSmithWebhookConfig(
                domain="agents.example.com",
                endpoint="api.example.com",
                backend_name="route53",
                webhook_secret="your-secret",
            ),
        )

    Then configure LangSmith to send project webhooks to:
        POST https://api.example.com/webhooks/langsmith
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _import_dns_aid() -> Any:
    """Lazy import dns_aid."""
    try:
        import dns_aid

        return dns_aid
    except ImportError:
        raise ImportError(
            "The dns-aid package is required for the LangSmith webhook bridge. "
            "Install it with: pip install dns-aid"
        )


def _import_create_backend() -> Any:
    """Lazy import dns_aid backend factory."""
    try:
        from dns_aid.backends import create_backend

        return create_backend
    except ImportError:
        raise ImportError(
            "The dns-aid package is required for the LangSmith webhook bridge. "
            "Install it with: pip install dns-aid"
        )


def _import_schema_adapter() -> Any:
    """Lazy import the schema registry adapter."""
    try:
        from dns_aid.core.schema_registry import from_langsmith_project

        return from_langsmith_project
    except ImportError:
        raise ImportError(
            "dns-aid>=0.13.0 is required for the LangSmith webhook bridge. "
            "Install it with: pip install 'dns-aid>=0.13.0'"
        )


# Events that trigger DNS publish/unpublish
PUBLISH_EVENTS = {"project.created", "project.updated"}
UNPUBLISH_EVENTS = {"project.archived", "project.deleted"}


@dataclass
class LangSmithWebhookConfig:
    """Configuration for the LangSmith webhook bridge.

    Attributes:
        domain: DNS domain to publish agents under.
        endpoint: Hostname where agents are reachable.
        port: Port number for the agent endpoint.
        backend_name: DNS backend name ('route53', 'cloudflare', 'ddns').
        backend: Pre-configured DNSBackend instance (overrides backend_name).
        webhook_secret: HMAC secret for verifying LangSmith webhook signatures.
            If None, signature verification is skipped (NOT recommended for production).
        webhook_path: URL path for the webhook endpoint.
        default_protocol: Default protocol for published agents.
        auto_publish: If True, publish on project.created/updated events.
        auto_unpublish: If True, unpublish on project.archived/deleted events.
        project_filter: Optional callable to filter which projects get published.
            Receives the project dict, returns True to publish.
    """

    domain: str = ""
    endpoint: str = ""
    port: int = 443
    backend_name: Optional[str] = None
    backend: Any = None
    webhook_secret: Optional[str] = None
    webhook_path: str = "/webhooks/langsmith"
    default_protocol: str = "https"
    auto_publish: bool = True
    auto_unpublish: bool = True
    project_filter: Any = None  # Callable[[dict], bool] | None


def _verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 webhook signature."""
    expected = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


def _get_backend(config: LangSmithWebhookConfig) -> Any:
    """Resolve DNS backend from config."""
    if config.backend is not None:
        return config.backend
    if config.backend_name:
        create_backend = _import_create_backend()
        return create_backend(config.backend_name)
    return None


async def handle_webhook_event(
    event: dict[str, Any],
    config: LangSmithWebhookConfig,
) -> dict[str, Any]:
    """Process a single LangSmith webhook event.

    Args:
        event: Webhook event payload with 'event' and 'data' keys.
        config: Webhook configuration.

    Returns:
        Result dict with 'action' and 'success' keys.
    """
    event_type = event.get("event", "")
    data = event.get("data", {})
    project = data.get("project", data)

    if not project.get("name"):
        logger.warning("Webhook event missing project name, skipping")
        return {"action": "skipped", "reason": "no project name"}

    # Apply project filter
    if config.project_filter and not config.project_filter(project):
        logger.debug("Project filtered out", project=project.get("name"))
        return {"action": "filtered", "project": project.get("name")}

    if event_type in PUBLISH_EVENTS and config.auto_publish:
        dns_aid = _import_dns_aid()
        from_langsmith = _import_schema_adapter()
        backend = _get_backend(config)

        publish_kwargs = from_langsmith(
            project,
            domain=config.domain,
            endpoint=config.endpoint,
            protocol=config.default_protocol,
            port=config.port,
        )

        try:
            result = await dns_aid.publish(
                **publish_kwargs,
                backend=backend,
            )
            logger.info(
                "LangSmith webhook: Published agent '%s'",
                publish_kwargs.get("name"),
            )
            return {
                "action": "published",
                "project": project.get("name"),
                "agent_name": publish_kwargs.get("name"),
                "success": result.success,
            }
        except Exception:
            logger.exception(
                "LangSmith webhook: Failed to publish '%s'",
                publish_kwargs.get("name"),
            )
            return {
                "action": "publish_failed",
                "project": project.get("name"),
                "success": False,
            }

    elif event_type in UNPUBLISH_EVENTS and config.auto_unpublish:
        dns_aid = _import_dns_aid()
        backend = _get_backend(config)

        name = project.get("name", "").lower().replace(" ", "-").replace("_", "-")
        name = "".join(c if c.isalnum() or c == "-" else "-" for c in name).strip("-")[:63]

        protocol = (project.get("metadata", {}) or {}).get(
            "protocol", config.default_protocol
        )

        try:
            deleted = await dns_aid.unpublish(
                name=name,
                domain=config.domain,
                protocol=protocol,
                backend=backend,
            )
            logger.info(
                "LangSmith webhook: Unpublished agent '%s' (deleted=%s)",
                name,
                deleted,
            )
            return {
                "action": "unpublished",
                "project": project.get("name"),
                "agent_name": name,
                "success": True,
                "deleted": deleted,
            }
        except Exception:
            logger.exception(
                "LangSmith webhook: Failed to unpublish '%s'", name
            )
            return {
                "action": "unpublish_failed",
                "project": project.get("name"),
                "success": False,
            }

    return {"action": "ignored", "event": event_type}


def add_langsmith_webhook(
    app: Any,
    config: LangSmithWebhookConfig,
) -> None:
    """Add the LangSmith webhook endpoint to a FastAPI app.

    Args:
        app: FastAPI application instance.
        config: Webhook bridge configuration.
    """
    try:
        from fastapi import FastAPI as FastAPIClass
        from fastapi import Header, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError(
            "FastAPI is required. Install it with: pip install 'langserve[server]'"
        )

    if not isinstance(app, FastAPIClass):
        raise TypeError(
            f"Expected a FastAPI app, got {type(app).__name__}"
        )

    @app.post(config.webhook_path)
    async def langsmith_webhook(
        request: Request,
        x_langsmith_signature: str | None = Header(default=None),
    ) -> JSONResponse:
        """Handle incoming LangSmith webhook events."""
        body = await request.body()

        # Verify signature if secret is configured
        if config.webhook_secret:
            if not x_langsmith_signature:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Missing X-LangSmith-Signature header"},
                )
            if not _verify_signature(body, x_langsmith_signature, config.webhook_secret):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid signature"},
                )

        try:
            import json

            event = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON"},
            )

        result = await handle_webhook_event(event, config)
        status = 200 if result.get("success", True) else 500
        return JSONResponse(status_code=status, content=result)

    logger.info(
        "LangSmith webhook bridge registered at %s", config.webhook_path
    )
