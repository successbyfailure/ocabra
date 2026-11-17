"""``python -m ocabra.docker_agent``: run the docker agent (see :mod:`.server`)."""

from __future__ import annotations

import os

import uvicorn

from ocabra.config import settings

from .ops import AGENT_ENV


def main() -> None:
    os.environ[AGENT_ENV] = "1"
    uvicorn.run(
        "ocabra.docker_agent.server:app",
        host="0.0.0.0",  # noqa: S104 - compose network only, the port is not published
        port=settings.docker_agent_port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
