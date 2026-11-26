"""SQLAlchemy models for cloud GPU nodes (temporary federation peers)."""

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

# cloud_nodes has FKs to federation_invites / federation_peers: register them
# with the metadata whenever this module is used on its own.
import ocabra.db.federation  # noqa: E402,F401
from ocabra.database import Base


class CloudProvider(Base):
    """Credentials and defaults for one provider (``runpod``)."""

    __tablename__ = "cloud_providers"

    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # Nullable in the migrated schema (server_default fills it); kept as is.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=True
    )


class CloudNode(Base):
    """A pod we launched. Becomes a federation peer once it joins with its invite."""

    __tablename__ = "cloud_nodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_pod_id: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # creating | starting | running | stopping | terminated | error
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="creating", index=True)
    gpu_type_id: Mapped[str] = mapped_column(String(128), nullable=False)
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    cloud_type: Mapped[str] = mapped_column(String(16), nullable=False, default="COMMUNITY")
    image: Mapped[str] = mapped_column(String(512), nullable=False)
    cost_per_hr: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_runtime_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    idle_shutdown_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    invite_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("federation_invites.id", ondelete="SET NULL"), nullable=True
    )
    peer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("federation_peers.id", ondelete="SET NULL"), nullable=True
    )
    preload_models: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    # Marcas de cada etapa hasta estar listo (core/cloud/timeline.py).
    timeline: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    # Network volume (model/backend cache) and the data center it pins the pod to.
    network_volume_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_center_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Nullable in the migrated schema (server_default fills it); kept as is.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    terminated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    terminate_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # functional | zero_data — what it was launched with (relaunch repeats it).
    privacy_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)


class CloudModelRecipe(Base):
    """How to give a cloud node one model: where it comes from and its config.

    Captured from the nodes themselves (a model downloaded and tuned by hand
    in a pod's panel is otherwise lost when the pod goes), and offered again
    when launching the next one. ``steps`` is the same plan the preload runs
    (core/cloud/preload.py): download (+ register) and load.
    """

    __tablename__ = "cloud_model_recipes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_model_id: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    backend_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str | None] = mapped_column(String(512), nullable=True)
    """Human summary of where it comes from (``huggingface: repo/file``)."""
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    steps: Mapped[list] = mapped_column(JSONB, nullable=False)
    last_node_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_gpu_type_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CloudLaunchRequest(Base):
    """Petición de máquina que se reintenta hasta que hay capacidad.

    Guarda el lanzamiento entero (``launch``, los argumentos de
    ``CloudNodeManager.launch`` sin la GPU) y una lista de GPUs aceptables por
    orden de preferencia. El bucle de nube la prueba cada
    ``retry_interval_s`` hasta que lanza un pod o caduca.
    """

    __tablename__ = "cloud_launch_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # waiting | launched | expired | cancelled | failed
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="waiting")
    gpu_type_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    launch: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    retry_interval_s: Mapped[int] = mapped_column(Integer, nullable=False, default=120)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cloud_node_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cloud_nodes.id", ondelete="SET NULL"), nullable=True
    )
    launched_gpu_type_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # manual | benchmark: quién la pidió (una ejecución de benchmark con pod propio).
    purpose: Mapped[str] = mapped_column(String(16), nullable=False, default="manual")
    benchmark_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
