"""The app's routes as FastAPI matches them, whatever the FastAPI version.

FastAPI 0.163 (the api image, see constraints.txt) keeps included routers as
``_IncludedRouter`` entries in ``app.routes`` instead of copying their routes
in, so ``app.routes`` no longer lists ``/ocabra/models/...``. Tests that
inspected it only ever passed against the older FastAPI of a local venv.
"""

from __future__ import annotations


def effective_routes(app) -> list[tuple[str, set[str]]]:
    """``(full path, methods)`` of every route, in matching order."""
    try:
        from fastapi.routing import iter_route_contexts
    except ImportError:  # FastAPI before the lazy include: routes are flat
        return [
            (route.path, set(getattr(route, "methods", None) or ()))
            for route in app.routes
            if hasattr(route, "path")
        ]
    return [(ctx.path, set(ctx.methods or ())) for ctx in iter_route_contexts(app.routes)]
