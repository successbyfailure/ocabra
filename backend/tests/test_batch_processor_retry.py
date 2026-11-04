from ocabra.core.batch_processor import _is_transient_batch_error


def test_insufficient_vram_409_is_transient() -> None:
    assert _is_transient_batch_error(409, {"code": "insufficient_vram"}) is True


def test_other_409_codes_stay_permanent() -> None:
    assert _is_transient_batch_error(409, {"code": "model_at_capacity"}) is False


def test_insufficient_vram_at_other_status_codes_stays_permanent() -> None:
    # Same code, but not the 409 this specific condition came from — don't
    # widen retry eligibility beyond what's been reproduced/verified.
    assert _is_transient_batch_error(500, {"code": "insufficient_vram"}) is False


def test_missing_code_stays_permanent() -> None:
    assert _is_transient_batch_error(409, {"message": "no code field"}) is False
