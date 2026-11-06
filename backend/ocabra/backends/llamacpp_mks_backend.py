"""Makespace fork of the llama.cpp backend (``llamacpp_mks``).

A separate ``backend_type`` from upstream ``llama_cpp`` on purpose, so both can
be installed at the same time and compared A/B, with per-model rollback by
pointing a profile's ``base_model_id`` at one or the other. Decided 2026-11-06;
see ``docs/tasks/backend-distribution-plan.md``.

Everything about running a model is inherited from :class:`LlamaCppBackend` —
the fork is upstream plus extra endpoints, not a different engine. Only the
install source and the identity differ. Consequently, any behaviour keyed off
"this is llama.cpp" must test ``ocabra.core.model_ref.LLAMA_CPP_FAMILY``
rather than ``== "llama_cpp"``, or it will silently skip this backend.

The fork currently carries on-demand vision: it can load and drop the
multimodal projector at runtime, without the process restart and full weight
reload that upstream requires. See the branch's commit for details.
"""

from __future__ import annotations

from ocabra.backends.base import BackendInstallSpec

from .llama_cpp_backend import LlamaCppBackend

#: Upstream of the fork, for documentation and the install spec.
FORK_REPO = "https://github.com/makespacemadrid/llama.cpp-mks"
FORK_REF = "ocabra/elastic-kv"


class LlamaCppMksBackend(LlamaCppBackend):
    BACKEND_TYPE = "llamacpp_mks"
    BACKEND_DIR_NAME = "llamacpp_mks"
    # ``llama_cpp_custom_server_bin`` is upstream's pre-fork workaround for the
    # install path not being able to build a custom binary. This backend builds
    # its own, so honouring that setting here would point both backends at the
    # same file.
    HONOUR_CUSTOM_SERVER_BIN_SETTING = False
    # No fat-image fallback: that path holds the upstream binary, and silently
    # serving it would make this backend indistinguishable from llama_cpp while
    # reporting fork capabilities it does not have.
    LEGACY_SERVER_BIN_SETTING = None

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Same native cmake build as upstream, from the fork's branch.

        Reuses ``install_llama_cpp.sh`` verbatim: the build recipe is
        identical, only the checked-out source differs. Installing updates the
        backend too — re-running it with a moved ``git_ref`` is the fork's
        deployment mechanism, since there is no separate update endpoint yet.
        """
        upstream = super().install_spec
        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-llama-cpp-mks",
            oci_tags=dict(upstream.oci_tags),
            apt_packages=list(upstream.apt_packages),
            git_repo=FORK_REPO,
            git_ref=FORK_REF,
            post_install_script=upstream.post_install_script,
            extra_bins=dict(upstream.extra_bins),
            include_core_runtime=upstream.include_core_runtime,
            estimated_size_mb=upstream.estimated_size_mb,
            display_name="llama.cpp (Makespace)",
            description=(
                "Makespace fork of llama.cpp. Upstream plus on-demand vision: "
                "loads and drops the multimodal projector at runtime without "
                "restarting the server or reloading the model weights."
            ),
            tags=[*upstream.tags, "fork"],
        )
