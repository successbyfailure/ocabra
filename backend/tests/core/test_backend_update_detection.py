"""Update detection for OCI backend artifacts."""

from ocabra.api.internal.backends import _published_update_available
from ocabra.core.backend_installer import BackendInstallStatus, BackendModuleState


def _state(**changes: object) -> BackendModuleState:
    values = {
        "backend_type": "llamacpp_mks",
        "display_name": "llama.cpp MKS",
        "description": "",
        "tags": ["llm"],
        "install_status": BackendInstallStatus.INSTALLED,
        "install_source": "source",
        "source_commit": "da7f914bf0000000000000000000000000000000",
    }
    values.update(changes)
    return BackendModuleState(**values)  # type: ignore[arg-type]


def test_native_source_update_compares_source_commit() -> None:
    published = {
        "digest": "sha256:new",
        "packages": {},
        "version": "efbf556487",
    }

    assert _published_update_available(_state(), published) is True
    assert (
        _published_update_available(
            _state(source_commit="efbf556487d3f28141f12f4a9aa8a186f02d5753"),
            published,
        )
        is False
    )


def test_oci_update_still_compares_image_digest() -> None:
    published = {"digest": "sha256:new", "packages": {}, "version": "efbf556487"}

    assert (
        _published_update_available(
            _state(install_source="oci", oci_digest="sha256:old"), published
        )
        is True
    )
