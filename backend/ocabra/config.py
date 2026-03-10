from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    database_url: str = "postgresql+asyncpg://ocabra:ocabra@postgres:5432/ocabra"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Models storage
    models_dir: str = "/data/models"
    hf_cache_dir: str = "/data/hf_cache"
    hf_token: str = ""

    # GPU scheduling
    default_gpu_index: int = 1
    worker_port_range_start: int = 18000
    worker_port_range_end: int = 19000
    vram_buffer_mb: int = 512
    vram_pressure_threshold_pct: float = 90.0

    # Model lifecycle
    idle_timeout_seconds: int = 300

    # LiteLLM
    litellm_base_url: str = "http://litellm:4000"
    litellm_admin_key: str = ""
    litellm_auto_sync: bool = False

    # Energy
    energy_cost_eur_kwh: float = 0.15


settings = Settings()
