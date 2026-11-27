"""Pydantic schemas for cloud GPU nodes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CloudSettingsUpdate(BaseModel):
    """Provider defaults. Unset fields keep their value."""

    model_config = ConfigDict(extra="forbid")

    image: str | None = Field(default=None, min_length=3, max_length=512)
    cloud_type: str | None = Field(default=None, pattern=r"^(COMMUNITY|SECURE)$")
    default_ttl_minutes: int | None = Field(default=None, ge=0, le=7 * 24 * 60)
    default_idle_minutes: int | None = Field(default=None, ge=0, le=24 * 60)
    monthly_budget_usd: float | None = Field(default=None, ge=0, le=100_000)
    container_disk_gb: int | None = Field(default=None, ge=20, le=2000)
    pass_hf_token: bool | None = None
    hub_url: str | None = Field(default=None, max_length=1024)


class CloudProviderUpdate(BaseModel):
    """Set the API key (validated against the provider) and/or defaults.

    ``api_key=""`` deletes the stored key.
    """

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = Field(default=None, max_length=512)
    settings: CloudSettingsUpdate | None = None


class CloudNodeLaunch(BaseModel):
    """Launch a pod that joins this node's federation as a swarm peer."""

    model_config = ConfigDict(extra="forbid")

    gpu_type_id: str = Field(..., min_length=1, max_length=128)
    gpu_count: int = Field(default=1, ge=1, le=8)
    cloud_type: str | None = Field(default=None, pattern=r"^(COMMUNITY|SECURE)$")
    max_runtime_minutes: int | None = Field(
        default=None, ge=0, le=7 * 24 * 60, description="0 = no limit; None = provider default"
    )
    idle_shutdown_minutes: int | None = Field(
        default=None, ge=0, le=24 * 60, description="0 = never; None = provider default"
    )
    name: str | None = Field(default=None, max_length=40, pattern=r"^[A-Za-z0-9 _.-]*$")
    container_disk_gb: int | None = Field(
        default=None, ge=20, le=2000, description="Disco del pod en GB; None = el de los ajustes"
    )
    models: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Our model ids to download and load on the node once it joins",
    )
    recipes: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Recipe ids (GET /cloud/recipes): models captured from earlier pods",
    )
    backends: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Backends to install on the node when it joins, before any model. The "
            "ones the chosen models need are added anyway."
        ),
    )
    download_only: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Ids from models/recipes to download but not load (loaded when asked for)",
    )
    privacy_mode: str = Field(
        default="functional",
        pattern=r"^(functional|zero_data)$",
        description=(
            "zero_data: the node keeps nothing about requests (core/privacy.py); forces "
            "Secure Cloud and never receives the HF token"
        ),
    )
    volume_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Network volume to mount as the node's model/backend cache. Pins the pod "
            "to the volume's data center and Secure Cloud."
        ),
    )


class CloudVolumeCreate(BaseModel):
    """A network volume for models and backends."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="ocabra-models", min_length=1, max_length=60)
    size_gb: int = Field(..., ge=10, le=4096)
    data_center: str = Field(..., min_length=1, max_length=64)


class CloudVolumeResize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size_gb: int = Field(..., ge=10, le=4096, description="New size; volumes only grow")


class CloudNodeModels(BaseModel):
    """Models to add to a running cloud node."""

    model_config = ConfigDict(extra="forbid")

    models: list[str] = Field(default_factory=list, max_length=10)
    recipes: list[str] = Field(default_factory=list, max_length=10)
    download_only: list[str] = Field(default_factory=list, max_length=20)
    backends: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _at_least_one(self) -> CloudNodeModels:
        if not self.models and not self.recipes and not self.backends:
            raise ValueError("Elige al menos un modelo o un backend")
        return self


class CloudNodeImageUpdate(BaseModel):
    """Restart a running cloud node on another ocabra-node image."""

    model_config = ConfigDict(extra="forbid")

    version: str | None = Field(
        default=None,
        pattern=r"^\d+\.\d+\.\d+$",
        description="Image tag X.Y.Z; empty = the configured image (usually latest)",
    )


class CloudRecipeCreate(BaseModel):
    """A llama.cpp model picked from Hugging Face or Ollama for cloud nodes."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., pattern=r"^(huggingface|ollama)$")
    model_ref: str = Field(..., min_length=1, max_length=255, description="HF repo or Ollama tag")
    artifact: str | None = Field(
        default=None, max_length=255, description="HF: the .gguf file in the repo"
    )
    display_name: str | None = Field(default=None, max_length=255)
    size_bytes: int | None = Field(default=None, ge=0)
    context: int | None = Field(
        default=None, ge=0, le=4_194_304, description="0/None = node default"
    )
    parallel: int | None = Field(default=None, ge=0, le=64)
    load_policy: str = Field(default="on_demand", pattern=r"^(on_demand|warm|pin)$")
    backend_type: str = Field(
        default="llamacpp_mks",
        pattern=r"^[a-z0-9_]+$",
        max_length=64,
        description="Backend the node serves it with (installed first if needed)",
    )

    @model_validator(mode="after")
    def _consistent(self) -> CloudRecipeCreate:
        llama = self.backend_type in ("llamacpp_mks", "llama_cpp", "bitnet")
        if self.backend_type == "ollama":
            raise ValueError("Los nodos en la nube no tienen Ollama: usa llama.cpp")
        if self.source == "ollama" and not llama:
            raise ValueError("Un tag de Ollama se sirve con llama.cpp")
        if (
            self.source == "huggingface"
            and llama
            and not (self.artifact or "").lower().endswith(".gguf")
        ):
            raise ValueError("Para llama.cpp, elige un fichero .gguf del repo")
        return self


class CloudRecipeUpdate(BaseModel):
    """Change a recipe before launching."""

    model_config = ConfigDict(extra="forbid")

    context: int | None = Field(default=None, ge=0, le=4_194_304, description="0 = node default")
    parallel: int | None = Field(default=None, ge=0, le=64, description="0 = node default")
    load_policy: str | None = Field(default=None, pattern=r"^(on_demand|warm|pin)$")
    display_name: str | None = Field(default=None, max_length=255)


class CloudLaunchRequestCreate(CloudNodeLaunch):
    """Pide una máquina y reintenta hasta que haya capacidad.

    Los mismos campos que un lanzamiento, más las GPUs aceptables por orden de
    preferencia (``gpu_type_id`` es la primera; ``alternatives`` el resto).
    """

    alternatives: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Otras GPUs aceptables, por orden de preferencia, si la primera no está",
    )
    wait_minutes: int = Field(
        default=360, ge=1, le=7 * 24 * 60, description="Cuánto seguir intentándolo"
    )
    retry_interval_minutes: int = Field(
        default=2, ge=1, le=120, description="Cada cuánto se vuelve a probar"
    )
