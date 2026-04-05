"""
POST /v1/pooling, /v1/score, /v1/rerank and /v1/classify.
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Request

from ocabra.api._deps_auth import UserContext

from ._deps import (
    _openai_error,
    check_capability,
    ensure_loaded,
    get_model_manager,
    get_openai_user,
    raise_upstream_http_error,
    to_backend_body,
)

router = APIRouter()


def _normalize_non_empty_text(
    value: Any, *, param: str, code: str, message: str
) -> str:
    if isinstance(value, str) and value.strip():
        return value
    raise _openai_error(
        message,
        "invalid_request_error",
        param=param,
        code=code,
    )


def _normalize_non_empty_text_list(
    value: Any, *, param: str, code: str, message: str
) -> list[str]:
    if isinstance(value, list) and value and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return value
    raise _openai_error(
        message,
        "invalid_request_error",
        param=param,
        code=code,
    )


def _normalize_classification_input(body: dict[str, Any]) -> dict[str, Any]:
    model_id = str(body.get("model", "")).strip()
    input_value = body.get("input")
    if not model_id:
        raise _openai_error(
            "The 'model' field is required.",
            "invalid_request_error",
            param="model",
            code="missing_model",
        )
    if isinstance(input_value, str):
        if not input_value.strip():
            raise _openai_error(
                "The 'input' field must not be empty.",
                "invalid_request_error",
                param="input",
                code="invalid_input",
            )
        body["model"] = model_id
        body["input"] = input_value
        return body
    if isinstance(input_value, list) and input_value and all(
        isinstance(item, str) and item.strip() for item in input_value
    ):
        body["model"] = model_id
        body["input"] = input_value
        return body
    raise _openai_error(
        "The 'input' field must be a non-empty string or a non-empty list of strings.",
        "invalid_request_error",
        param="input",
        code="invalid_input",
    )


def _normalize_score_request(body: dict[str, Any]) -> dict[str, Any]:
    model_id = str(body.get("model", "")).strip()
    queries = body.get("queries", body.get("query", body.get("text_1")))
    documents = body.get("documents", body.get("document", body.get("text_2")))
    if not model_id:
        raise _openai_error(
            "The 'model' field is required.",
            "invalid_request_error",
            param="model",
            code="missing_model",
        )

    if isinstance(queries, str) and isinstance(documents, str):
        body["model"] = model_id
        body["queries"] = _normalize_non_empty_text(
            queries,
            param="queries",
            code="invalid_queries",
            message="The 'queries' field must be a non-empty string or a non-empty list of strings.",
        )
        body["documents"] = _normalize_non_empty_text(
            documents,
            param="documents",
            code="invalid_documents",
            message="The 'documents' field must be a non-empty string or a non-empty list of strings.",
        )
        body.pop("query", None)
        body.pop("document", None)
        body.pop("text_1", None)
        body.pop("text_2", None)
        return body

    if isinstance(queries, list) and isinstance(documents, list):
        normalized_queries = _normalize_non_empty_text_list(
            queries,
            param="queries",
            code="invalid_queries",
            message="The 'queries' field must be a non-empty string or a non-empty list of strings.",
        )
        normalized_documents = _normalize_non_empty_text_list(
            documents,
            param="documents",
            code="invalid_documents",
            message="The 'documents' field must be a non-empty string or a non-empty list of strings.",
        )
        if len(normalized_queries) != len(normalized_documents):
            raise _openai_error(
                "Batch score requests require 'queries' and 'documents' to have the same length.",
                "invalid_request_error",
                param="documents",
                code="mismatched_batch_length",
            )
        body["model"] = model_id
        body["queries"] = normalized_queries
        body["documents"] = normalized_documents
        body.pop("query", None)
        body.pop("document", None)
        body.pop("text_1", None)
        body.pop("text_2", None)
        return body

    raise _openai_error(
        "Score requests require 'queries' and 'documents' as either strings or lists of equal length.",
        "invalid_request_error",
        param="queries",
        code="invalid_score_shape",
    )


def _normalize_rerank_documents(documents: list[Any]) -> list[str]:
    normalized: list[str] = []
    for index, item in enumerate(documents):
        if isinstance(item, str) and item.strip():
            normalized.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                normalized.append(text)
                continue
        raise _openai_error(
            f"Document at index {index} must be a non-empty string or an object with a non-empty 'text' field.",
            "invalid_request_error",
            param="documents",
            code="invalid_documents",
        )
    return normalized


def _normalize_rerank_request(body: dict[str, Any]) -> dict[str, Any]:
    model_id = str(body.get("model", "")).strip()
    query = body.get("query")
    documents = body.get("documents")
    top_n = body.get("top_n")
    if not model_id:
        raise _openai_error(
            "The 'model' field is required.",
            "invalid_request_error",
            param="model",
            code="missing_model",
        )
    if not isinstance(query, str) or not query.strip():
        raise _openai_error(
            "The 'query' field must be a non-empty string.",
            "invalid_request_error",
            param="query",
            code="invalid_query",
        )
    if not isinstance(documents, list) or not documents:
        raise _openai_error(
            "The 'documents' field must be a non-empty list.",
            "invalid_request_error",
            param="documents",
            code="invalid_documents",
        )
    body["model"] = model_id
    body["query"] = query
    body["documents"] = _normalize_rerank_documents(documents)
    if top_n is not None:
        if not isinstance(top_n, int) or top_n < 1:
            raise _openai_error(
                "The 'top_n' field must be a positive integer.",
                "invalid_request_error",
                param="top_n",
                code="invalid_top_n",
            )
        if top_n > len(body["documents"]):
            raise _openai_error(
                "The 'top_n' field must be less than or equal to the number of documents.",
                "invalid_request_error",
                param="top_n",
                code="invalid_top_n",
            )
    return body


@router.post("/pooling", summary="Run pooling on a model")
async def pooling(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Run pooling/embedding-style inference on a model configured with pooling support.
    """
    body = await request.json()
    model_id: str = body.get("model", "")

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id, user=user)
    check_capability(state, "pooling", "pooling")
    model_id = state.model_id

    worker_pool = request.app.state.worker_pool
    try:
        return await worker_pool.forward_request(model_id, "/pooling", to_backend_body(state, body))
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)


@router.post("/score", summary="Score text pairs")
async def score(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Score pairs with a model exposing similarity/score support.
    """
    body = _normalize_score_request(await request.json())
    model_id: str = body.get("model", "")

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id, user=user)
    check_capability(state, "score", "score")
    model_id = state.model_id

    worker_pool = request.app.state.worker_pool
    try:
        return await worker_pool.forward_request(model_id, "/score", to_backend_body(state, body))
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)


@router.post("/rerank", summary="Rerank documents for a query")
async def rerank(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Rerank candidate documents against a query using a reranker-capable model.
    """
    body = _normalize_rerank_request(await request.json())
    model_id: str = body.get("model", "")

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id, user=user)
    check_capability(state, "rerank", "rerank")
    model_id = state.model_id

    worker_pool = request.app.state.worker_pool
    try:
        return await worker_pool.forward_request(model_id, "/rerank", to_backend_body(state, body))
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)


@router.post("/classify", summary="Classify inputs with a classification model")
async def classify(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Run text classification on one or many inputs.
    """
    body = _normalize_classification_input(await request.json())
    model_id: str = body.get("model", "")

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id, user=user)
    check_capability(state, "classification", "classification")
    model_id = state.model_id

    worker_pool = request.app.state.worker_pool
    try:
        return await worker_pool.forward_request(model_id, "/classify", to_backend_body(state, body))
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
