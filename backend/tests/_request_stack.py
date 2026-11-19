"""The request middlewares as ``ocabra.main`` registers them, for test apps."""

from __future__ import annotations

from fastapi import FastAPI

from ocabra.core.admission import AdmissionMiddleware
from ocabra.stats.collector import StatsMiddleware, record_admission_rejection


def install_request_stack(app: FastAPI) -> None:
    """Stats inside, admission just outside it (see ``ocabra/main.py``)."""
    app.add_middleware(StatsMiddleware)
    app.add_middleware(AdmissionMiddleware, on_reject=record_admission_rejection)
