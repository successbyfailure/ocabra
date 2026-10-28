from ocabra.core.vram_capacity import apply_use_case_flags


def test_llama_cpp_ctx_size_multiplies_back_out_by_slots():
    """plan['effective_context'] is the PER-SLOT context (vram_planner divides
    the fitting budget by slots to compute it), but llama-server's --ctx-size
    is the TOTAL pool it then divides by --parallel internally. Passing the
    per-slot value straight through used to double-divide it."""
    plan = {"applied": True, "effective_context": 32768, "slots": 2}
    merged = apply_use_case_flags({}, "llama_cpp", plan)
    assert merged["llama_cpp"]["ctx_size"] == 65536
    assert merged["llama_cpp"]["parallel_slots"] == 2


def test_llama_cpp_ctx_size_unchanged_for_single_slot():
    plan = {"applied": True, "effective_context": 65536, "slots": 1}
    merged = apply_use_case_flags({}, "llama_cpp", plan)
    assert merged["llama_cpp"]["ctx_size"] == 65536
    assert merged["llama_cpp"]["parallel_slots"] == 1


def test_bitnet_gets_the_same_multiplication():
    plan = {"applied": True, "effective_context": 4096, "slots": 4}
    merged = apply_use_case_flags({}, "bitnet", plan)
    assert merged["llama_cpp"]["ctx_size"] == 16384


def test_vllm_max_model_len_is_not_multiplied_by_slots():
    """vLLM's paged KV pool is sized by gpu_memory_utilization up front, not
    by ctx * parallel — max_model_len is already the right per-request value,
    unlike llama.cpp's ctx-size."""
    plan = {"applied": True, "effective_context": 32768, "slots": 4}
    merged = apply_use_case_flags({}, "vllm", plan)
    assert merged["vllm"]["max_model_len"] == 32768


def test_ollama_num_ctx_is_not_multiplied_by_slots():
    """Ollama's num_ctx is already per-slot semantics (it reserves
    num_ctx * NUM_PARALLEL internally) — no extra multiplication needed."""
    plan = {"applied": True, "effective_context": 32768, "slots": 2}
    merged = apply_use_case_flags({}, "ollama", plan)
    assert merged["ollama"]["num_ctx"] == 32768
    assert merged["ollama"]["num_parallel"] == 2


def test_returns_extra_config_unchanged_when_not_applied():
    plan = {"applied": False, "effective_context": 32768, "slots": 2}
    original = {"llama_cpp": {"ctx_size": 1234}}
    merged = apply_use_case_flags(original, "llama_cpp", plan)
    assert merged == original
