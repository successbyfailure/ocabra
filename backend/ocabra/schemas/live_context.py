"""Request body for changing a loaded llama.cpp worker's context in place."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LiveContextRequest(BaseModel):
    """Sizes for ``POST /ocabra/models/{model_id}/context``.

    Fields left out keep their current value. ``restore`` returns the worker to
    the sizes it was loaded with and cannot be combined with the others.
    """

    model_config = ConfigDict(extra="forbid")

    context: int | None = Field(
        default=None, gt=0, description="KV pool in tokens (llama-server --ctx-size)."
    )
    parallel: int | None = Field(default=None, gt=0, le=64, description="Slots.")
    ubatch: int | None = Field(default=None, gt=0, le=8192, description="Physical batch.")
    batch: int | None = Field(default=None, gt=0, le=65536, description="Logical batch.")
    restore: bool = Field(default=False, description="Go back to the loaded sizes.")

    @model_validator(mode="after")
    def _one_kind_of_change(self) -> LiveContextRequest:
        sizes = (self.context, self.parallel, self.ubatch, self.batch)
        if self.restore and any(value is not None for value in sizes):
            raise ValueError("restore cannot be combined with explicit sizes")
        if not self.restore and all(value is None for value in sizes):
            raise ValueError("give at least one of context, parallel, ubatch, batch, or restore")
        return self
