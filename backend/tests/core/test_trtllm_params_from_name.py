"""El tamano de modelo se extrae del nombre como numero, no por subcadena.

La version anterior era una cadena de ``elif "<n>b" in name`` evaluada en orden,
y ganaba la primera coincidencia. Como ``"4b"`` es subcadena de ``"14b"`` y
``"7b"`` de ``"27b"``, los tamanos de dos digitos caian en la rama de un digito:
un 14B se estimaba como 4B y un 27B como 7B. Esta funcion calcula la VRAM para
compilar motores TRT-LLM, asi que infravalorar 4x acababa en OOM.
"""

import pytest

from ocabra.api.internal.trtllm import _params_b_from_name


@pytest.mark.parametrize(
    ("nombre", "esperado"),
    [
        # Los tamanos que la cadena de elif ya acertaba siguen igual.
        ("Qwen2.5-0.5B", 0.5),
        ("Qwen3-0.6B", 0.6),
        ("TinyLlama-1.1B", 1.1),
        ("Qwen2.5-1.5B", 1.5),
        ("Llama-3.2-3B", 3.0),
        ("gemma-3-4b-it", 4.0),
        ("Mistral-7B-v0.3", 7.0),
        ("Meta-Llama-3.1-8B-Instruct", 8.0),
        ("gemma-2-9b", 9.0),
        ("gemma-3-12b-it", 12.0),
        ("Qwen3-32B", 32.0),
        ("Llama-3.3-70B", 70.0),
        ("Qwen2.5-72B", 72.0),
    ],
)
def test_tamanos_que_ya_funcionaban_no_cambian(nombre, esperado):
    assert _params_b_from_name(nombre) == esperado


@pytest.mark.parametrize(
    ("nombre", "esperado", "antes_daba"),
    [
        ("Qwen2.5-14B-Instruct", 14.0, 4.0),   # "4b" dentro de "14b"
        ("gemma-3-27b-it", 27.0, 7.0),         # "7b" dentro de "27b"
        ("Llama-2-13B-chat", 13.0, 3.0),       # "3b" dentro de "13b"
        ("Yi-1.5-34B", 34.0, 4.0),             # "4b" dentro de "34b"
        ("Qwen3-30B-A3B", 30.0, 3.0),          # "3b" dentro de "30b"
        ("Qwen3-235B-A22B", 235.0, 7.0),       # no estaba en la cadena
    ],
)
def test_tamanos_que_la_cadena_de_elif_fallaba(nombre, esperado, antes_daba):
    assert _params_b_from_name(nombre) == esperado
    assert esperado != antes_daba


@pytest.mark.parametrize(
    "nombre",
    ["nombre-sin-tamano", "Qwen2.5", "", "modelo-int8", "awq-gptq"],
)
def test_sin_tamano_en_el_nombre_devuelve_none(nombre):
    assert _params_b_from_name(nombre) is None


def test_no_confunde_una_b_pegada_a_otras_letras():
    # "bf16" lleva un digito seguido de letras: no es un tamano.
    assert _params_b_from_name("modelo-bf16") is None


def test_descarta_numeros_fuera_de_rango():
    assert _params_b_from_name("raro-99999b") is None
