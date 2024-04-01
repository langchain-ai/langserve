"""Test the playground API."""

from fastapi import APIRouter, FastAPI
from httpx import AsyncClient
from langchain_core.runnables import RunnableLambda

from langserve import add_routes


async def test_serve_playground() -> None:
    """Test the server directly via HTTP requests."""
    app = FastAPI()
    add_routes(
        app,
        RunnableLambda(lambda foo: "hello"),
    )

    async with AsyncClient(app=app, base_url="http://localhost:9999") as client:
        response = await client.get("/playground/index.html")
        assert response.status_code == 200
        # Test that we can't access files that do not exist.
        response = await client.get("/playground/i_do_not_exist.txt")
        assert response.status_code == 404
        # Test that we can't access files outside of the playground directory
        response = await client.get("/playground//etc/passwd")
        assert response.status_code == 404


async def test_serve_playground_with_api_router() -> None:
    """Test serving playground from an api router with a prefix."""
    app = FastAPI()

    # Make sure that we can add routers
    # to an API router
    router = APIRouter(prefix="/langserve_runnables")

    add_routes(
        router,
        RunnableLambda(lambda foo: "hello"),
        path="/chat",
    )

    app.include_router(router)

    async with AsyncClient(app=app, base_url="http://localhost:9999") as client:
        response = await client.get("/langserve_runnables/chat/playground/index.html")
        assert response.status_code == 200


async def test_serve_playground_with_api_router_in_api_router() -> None:
    """Test serving playground from an api router within an api router."""
    app = FastAPI()

    router = APIRouter(prefix="/foo")

    add_routes(
        router,
        RunnableLambda(lambda foo: "hello"),
    )

    parent_router = APIRouter(prefix="/parent")
    parent_router.include_router(router, prefix="/bar")

    # Now add parent router to the app
    app.include_router(parent_router)

    async with AsyncClient(app=app, base_url="http://localhost:9999") as client:
        response = await client.get("/parent/bar/foo/playground/index.html")
        assert response.status_code == 200


async def test_root_path_on_playground() -> None:
    """Test that the playground respects the root_path for requesting assets"""

    for root_path in ("/home/root", "/home/root/"):
        app = FastAPI(root_path=root_path)
        add_routes(
            app,
            RunnableLambda(lambda foo: "hello"),
            path="/chat",
        )

        router = APIRouter(prefix="/router")
        add_routes(
            router,
            RunnableLambda(lambda foo: "hello"),
            path="/chat",
        )
        app.include_router(router)

        async_client = AsyncClient(app=app, base_url="http://localhost:9999")

        response = await async_client.get("/chat/playground/index.html")
        assert response.status_code == 200
        assert (
            f'src="{root_path.rstrip("/")}/chat/playground/assets/'
            in response.content.decode()
        ), "html should contain reference to playground assets with root_path prefix"

        response = await async_client.get("/router/chat/playground/index.html")
        assert response.status_code == 200
        assert (
            f'src="{root_path.rstrip("/")}/router/chat/playground/assets/'
            in response.content.decode()
        ), "html should contain reference to playground assets with root_path prefix"
