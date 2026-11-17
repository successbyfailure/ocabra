"""Docker agent: the only process meant to hold the Docker socket (SEG-03).

See :mod:`.ops` for the allowed operations and :mod:`.server` for the HTTP
side. The api talks to it through :mod:`ocabra.core.docker_ops`.
"""
