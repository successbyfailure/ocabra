import re

from ocabra.main import _build_cors_config


def test_cors_allows_loopback_dev_origin() -> None:
    config = _build_cors_config()
    pattern = re.compile(str(config["allow_origin_regex"]))

    assert pattern.fullmatch("http://localhost:5173")
    assert pattern.fullmatch("https://127.0.0.1:4173")


def test_cors_rejects_unexpected_origin() -> None:
    config = _build_cors_config()
    pattern = re.compile(str(config["allow_origin_regex"]))

    assert pattern.fullmatch("https://evil.example") is None
