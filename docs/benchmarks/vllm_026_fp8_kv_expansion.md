# Expansión de contexto con KV FP8 en RTX 3090

Fecha: 2026-07-26

## Resultado

Se evaluó KV FP8 E4M3 con escalas dinámicas sobre los demás modelos de la
ronda vLLM 0.26. Todos los checkpoints declaran 262.144 tokens nativos.

| Modelo | Contexto anterior | Contexto final | Validación larga |
|---|---:|---:|---:|
| Nemotron 3 Nano 30B-A3B | 65.536 | 262.144 | 240.020 tokens en 74,73 s |
| Qwen3-Coder 30B-A3B | 65.536 | 114.688 | 105.012 tokens en 74,60 s |
| Qwen3.6 35B-A3B | 32.768 | 65.536 | 60.014 tokens en 13,58 s |
| Gemma 4 26B-A4B | 65.536 BF16 | 65.536 BF16 | FP8 no compatible |
| Gemma 4 12B | 262.144 BF16 | 262.144 BF16 | Ya alcanza el límite nativo |

Los perfiles públicos registrados son:

- `gemma4:12b-vllm-ctx256k`
- `gemma4:26b-vllm-ctx64k`
- `nemotron-3-nano:30b-vllm-ctx256k`
- `qwen3-coder:30b-vllm-ctx112k`
- `qwen3.5:27b-vllm-ctx64k`
- `qwen3.6:35b-vllm-ctx64k`

Los `model_id` internos conservan los repositorios originales de Hugging Face.

## Configuraciones desplegadas

### Nemotron

- RTX 3090, TP=1 y contexto 262.144.
- KV FP8, FlashInfer y cálculo dinámico de escalas.
- KV fijada en 4,1 GiB (`--kv-cache-memory-bytes 4402341479`).
- `max_num_seqs=16`, batch de prefill 8.192.

El perfil automático reportó capacidad para 317.901 tokens. El contexto final
queda limitado por los 262.144 tokens nativos del checkpoint.

### Qwen3-Coder

- RTX 3090, TP=1 y contexto 114.688 (`ctx112k`).
- KV FP8, FlashInfer y cálculo dinámico de escalas.
- KV fijada en 5,4 GiB (`--kv-cache-memory-bytes 5798205850`).
- `max_num_seqs=16`, batch de prefill 8.192.

El techo automático fue 127.440 tokens. Se eligieron 112k para conservar
aproximadamente 1 GiB operativo para activaciones y CUDA.

### Qwen3.6

- RTX 3090, TP=1 y contexto 65.536.
- KV FP8, FlashInfer y cálculo dinámico de escalas.
- KV fijada en 0,8 GiB (`--kv-cache-memory-bytes 858993459`).
- `max_num_seqs=8`, batch de prefill 4.096.

El techo automático fue 81.744 tokens. Con batch 8.192, una entrada de 60k
agotaba 100 MiB en el bloque de atención lineal. Reducir el chunk a 4.096
eliminó ese pico y completó la misma entrada.

## Gemma 4

KV FP8 no se desplegó en los Gemma:

- FlashInfer rechaza el patrón `partial multimodal token full attention`.
- La selección automática usa Triton, cuyo KV FP8 requiere SM89 o superior.
- La RTX 3090 es SM86.

Gemma 4 12B mantiene sus 262.144 tokens nativos con KV BF16. Gemma 4 26B
mantiene 65.536 tokens con KV BF16; no se cambió a una configuración que no
pueda arrancar.
