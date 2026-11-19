"""Response shapes of the OpenAI-compatible Batches API (``/v1/batches``).

The objects are OpenAI's ``Batch`` and ``BatchList``; the routes returned
untyped dicts, so the published OpenAPI said ``{"type": "object"}`` and a
client generated from it could not read ``output_file_id``, ``error_file_id``,
``request_counts`` or the states. Extra keys are allowed, as OpenAI may add
fields; ``user`` and ``ocabra`` are oCabra extensions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BatchStatus = Literal[
    "validating",
    "failed",
    "in_progress",
    "finalizing",
    "completed",
    "expired",
    "cancelling",
    "cancelled",
]


class BatchRequestCounts(BaseModel):
    total: int = Field(..., description="Requests in the input file.")
    completed: int = Field(..., description="Requests finished (written to the output file).")
    failed: int = Field(..., description="Requests that failed (written to the error file).")


class BatchErrorItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str | None = None
    message: str | None = None
    param: str | None = None
    line: int | None = Field(None, description="1-based line of the input file.")


class BatchErrors(BaseModel):
    """Why the batch failed while being processed (``status`` = ``failed``).

    Errors in the input file are not here: ``POST /v1/batches`` refuses the
    file with a 400 that lists them (``error.errors``), and no batch is made.
    """

    model_config = ConfigDict(extra="allow")

    object: Literal["list"] = "list"
    data: list[BatchErrorItem] = Field(default_factory=list)


class BatchUser(BaseModel):
    id: str
    username: str | None = None


class BatchModelWait(BaseModel):
    model: str = Field(..., description="Model the waiting lines target.")
    status: str | None = Field(None, description="Its state now: loaded, loading, unloaded, error…")
    expected_wait_seconds: int | None = Field(
        None,
        description=(
            "While it loads: seconds left by its usual load time (median of its "
            "last loads); null without history. Same figure as the "
            "X-Ocabra-Expected-Wait-Seconds header of an interactive request."
        ),
    )


class BatchLiveStatus(BaseModel):
    """oCabra extension: why an unfinished batch is where it is.

    Present while the batch is ``validating``, ``in_progress``, ``finalizing``
    or ``cancelling``; ``null`` once it ends. Built at read time from the
    batch processor's live state.
    """

    phase: Literal[
        "queued",
        "starting",
        "loading_model",
        "running",
        "yielding_to_interactive",
        "waiting_for_vram",
        "finalizing",
        "cancelling",
    ] = Field(
        ...,
        description=(
            "queued: other batches go first (one runs at a time). starting: "
            "picked up, reading its input. loading_model: its lines wait for "
            "their model to load. running: lines are being served. "
            "yielding_to_interactive: its model was lent to interactive "
            "traffic and the lines retry when the cooldown ends. "
            "waiting_for_vram: lines got no GPU memory and retry shortly. "
            "finalizing: writing the output files. cancelling: stopping."
        ),
    )
    detail: str = Field(..., description="The same, as a sentence for a progress bar.")
    queue_position: int | None = Field(
        None, description="Batches ahead of this one (only while queued)."
    )
    in_flight: int = Field(0, description="Lines being dispatched right now.")
    waiting: int = Field(0, description="Lines waiting for a retry (cooldown or VRAM).")
    preemptions: int = Field(0, description="Lines interrupted so far for interactive traffic.")
    retry_in_seconds: int | None = Field(None, description="When waiting lines are retried next.")
    models: list[BatchModelWait] = Field(default_factory=list)


class Batch(BaseModel):
    """OpenAI ``Batch`` object."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., examples=["batch_3f1c…"])
    object: Literal["batch"] = "batch"
    endpoint: str
    errors: BatchErrors | dict[str, Any] | None = None
    input_file_id: str
    completion_window: str
    status: BatchStatus
    output_file_id: str | None = Field(
        None,
        description=(
            "JSONL with one line per successful request (custom_id, "
            "response.status_code, response.body). Read it with GET "
            "/v1/files/{id}/content."
        ),
    )
    error_file_id: str | None = Field(
        None, description="JSONL with one line per failed request (custom_id, error)."
    )
    created_at: int
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: BatchRequestCounts
    total_tokens: int | None = None
    metadata: dict[str, Any] | None = None
    user: BatchUser | None = Field(None, description="oCabra extension: who created it.")
    ocabra: BatchLiveStatus | None = Field(
        None, description="oCabra extension: live status of an unfinished batch."
    )


class BatchList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Batch]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
