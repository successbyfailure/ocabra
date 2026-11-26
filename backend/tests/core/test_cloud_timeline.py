"""Línea de tiempo de un pod: cuánto tarda cada etapa hasta estar listo."""

from datetime import UTC, datetime, timedelta

from ocabra.core.cloud.timeline import stamp, summary, track_items

T0 = datetime(2026, 11, 26, 20, 0, tzinfo=UTC)


def test_las_etapas_se_marcan_una_vez_y_se_resumen_en_segundos() -> None:
    t = {"requested_at": T0.isoformat()}
    t = stamp(t, "launched_at", T0 + timedelta(minutes=10))
    t = stamp(t, "launched_at", T0 + timedelta(minutes=99))  # no se pisa
    t = stamp(t, "running_at", T0 + timedelta(minutes=12))
    t = stamp(t, "joined_at", T0 + timedelta(minutes=15))
    items = [{"model_id": "backend:llamacpp_mks", "status": "installing"},
             {"model_id": "llamacpp_mks/gemma4:12b", "status": "pending"}]  # fmt: skip
    t = track_items(t, items, T0 + timedelta(minutes=16))
    items[0]["status"] = "ready"
    items[1]["status"] = "downloading"
    t = track_items(t, items, T0 + timedelta(minutes=18))
    assert "ready_at" not in t
    items[1]["status"] = "loaded"
    t = track_items(t, items, T0 + timedelta(minutes=25))
    s = summary(t)
    assert s["wait_s"] == 600 and s["boot_s"] == 120 and s["connect_s"] == 180
    assert s["prepare_s"] == 600 and s["total_s"] == 1500
    assert s["items"] == {"backend:llamacpp_mks": 120.0, "llamacpp_mks/gemma4:12b": 420.0}
    assert s["ready_with_errors"] is False


def test_sin_unirse_no_esta_listo_y_un_error_se_anota() -> None:
    items = [{"model_id": "m", "status": "error"}]
    t = track_items({"launched_at": T0.isoformat()}, items, T0 + timedelta(minutes=5))
    assert "ready_at" not in t  # nunca se unió
    t = stamp(t, "joined_at", T0 + timedelta(minutes=6))
    t = track_items(t, items, T0 + timedelta(minutes=7))
    assert summary(t)["ready_with_errors"] is True and summary(t)["prepare_s"] == 60
    assert summary({})["total_s"] is None
