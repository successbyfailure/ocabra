"""Per-model federation strategy: where a model's requests are served.

The nodes of a federation add up their models (local + what each peer
offers, warm or cold). Each model can carry a strategy for choosing among
the places that have it. Decided with the user on 2026-11-17 (see
docs/archive/federation-strategies-plan.md); strategies are added one at a
time, as use cases need them:

* ``auto`` — the default: warm beats cold, local bias (what federation did
  before strategies existed).
* ``local_only`` — never leaves this node. Replaces the per-rung «Ejecución:
  Solo local» (the rung was the wrong place: where a model runs is about the
  model, not about its context ladder).
* ``prefer`` — an ordered list of nodes (peer names/ids, or ``local``). The
  first one online that has the model serves it, **even cold** (it loads it
  there): «I want the embeddings on Mapache». ``fallback`` says what to do
  when none of them can: go on as ``auto`` (default) or refuse with a 503.

Stored as one ``server_config`` row (``federation_model_policies``: model id →
policy), cached in :data:`POLICIES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

POLICIES_KEY = "federation_model_policies"

STRATEGIES = ("auto", "local_only", "prefer")
LOCAL_NODE = "local"


@dataclass(frozen=True)
class FederationPolicy:
    strategy: str = "auto"
    nodes: tuple[str, ...] = field(default_factory=tuple)
    """``prefer``: nodes in order (peer name or id, or ``local``)."""
    fallback: bool = True
    """``prefer``: when none of *nodes* can serve, go on as ``auto``."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FederationPolicy:
        data = data or {}
        return cls(
            strategy=str(data.get("strategy") or "auto"),
            nodes=tuple(str(n) for n in data.get("nodes") or ()),
            fallback=bool(data.get("fallback", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "nodes": list(self.nodes), "fallback": self.fallback}

    @property
    def is_default(self) -> bool:
        return self.strategy == "auto"


AUTO = FederationPolicy()

# model id (canonical base_model_id) → policy. Loaded by FederationManager.
POLICIES: dict[str, FederationPolicy] = {}


def clean_policy(data: dict[str, Any]) -> FederationPolicy:
    """Validate a policy from the API; raises ``ValueError`` on a bad one."""
    strategy = str(data.get("strategy") or "auto")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}' (one of: {', '.join(STRATEGIES)})")
    nodes: list[str] = []
    for raw in data.get("nodes") or ():
        node = str(raw).strip()
        if node and node.lower() not in (n.lower() for n in nodes):
            nodes.append(LOCAL_NODE if node.lower() == LOCAL_NODE else node)
    if strategy == "prefer" and not nodes:
        raise ValueError("The 'prefer' strategy needs at least one node")
    if strategy != "prefer":
        nodes = []
    return FederationPolicy(
        strategy=strategy, nodes=tuple(nodes), fallback=bool(data.get("fallback", True))
    )


def policy_for(model_id: str | None) -> FederationPolicy:
    if not model_id:
        return AUTO
    return POLICIES.get(model_id, AUTO)


def node_matches(node: str, peer: Any) -> bool:
    """Whether the stored node name/id *node* designates *peer*."""
    wanted = node.strip().lower()
    names = {
        str(getattr(peer, "name", "") or "").lower(),
        str(getattr(peer, "peer_id", "") or "").lower(),
        str(getattr(peer, "remote_node_id", "") or "").lower(),
    }
    return wanted in names - {""}
