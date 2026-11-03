import base64
import math
import struct

from ocabra.api.openai.embeddings import _apply_requested_dimensions, _truncate_and_renormalize


def test_truncate_and_renormalize_shrinks_and_unit_normalizes():
    values = [3.0, 4.0, 0.0, 0.0]  # norm 5
    result = _truncate_and_renormalize(values, 2)
    assert result == [0.6, 0.8]
    assert math.isclose(math.sqrt(sum(v * v for v in result)), 1.0)


def test_truncate_and_renormalize_all_zero_vector_stays_zero():
    assert _truncate_and_renormalize([0.0, 0.0, 0.0], 2) == [0.0, 0.0]


def test_apply_requested_dimensions_truncates_list_embeddings_in_place():
    result = {"data": [{"embedding": [1.0, 0.0, 0.0, 0.0], "index": 0}]}
    _apply_requested_dimensions(result, 2)
    assert result["data"][0]["embedding"] == [1.0, 0.0]


def test_apply_requested_dimensions_is_noop_when_already_requested_length():
    result = {"data": [{"embedding": [1.0, 0.0], "index": 0}]}
    _apply_requested_dimensions(result, 2)
    assert result["data"][0]["embedding"] == [1.0, 0.0]


def test_apply_requested_dimensions_handles_base64_encoding_format():
    native = [1.0, 0.0, 0.0, 0.0]
    encoded = base64.b64encode(struct.pack(f"<{len(native)}f", *native)).decode("ascii")
    result = {"data": [{"embedding": encoded, "index": 0}]}
    _apply_requested_dimensions(result, 2)
    raw = base64.b64decode(result["data"][0]["embedding"])
    decoded = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    assert decoded == [1.0, 0.0]
