# Sustitución de Qwen3-Coder AWQ en vLLM 0.26

Fecha: 2026-07-27

## Motivo

El perfil `qwen3-coder:30b-vllm-ctx112k`, basado en
`stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ`, generaba secuencias corruptas
desde el primer token (`????`, caracteres CJK y repeticiones). La corrupción
se reprodujo directamente contra el worker vLLM, sin pasar por la API de
oCabra.

## Reemplazo

- Checkpoint: `cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit`
- Perfil público: `qwen3-coder:30b-vllm-ctx56k`
- GPU: RTX 3090, TP=1
- Contexto: 57.344 tokens (56k)
- KV cache: BF16
- `gpu_memory_utilization=0.98`
- `max_num_seqs=16`
- `max_num_batched_tokens=4096`
- prefix caching y chunked prefill activados
- FlashInfer no se fuerza
- speculative decoding desactivado

## Validación

Con una configuración conservadora de 32k y KV BF16:

- `Reply with exactly: OK` → `OK`
- `17 * 23` → `391`
- generación de una función Python sencilla → código correcto
- velocidad observada: 156–170 tokens/s con una petición

La prueba larga usó 51.805 tokens de entrada y una clave situada en el
registro 7 de un conjunto de 2.800 registros. El modelo recuperó correctamente
`OCABRA-57344` en 24,1 segundos.

## Techo de contexto

Con KV BF16 y `gpu_memory_utilization=0.95`, vLLM calculó:

- KV disponible: 4,98 GiB
- máximo matemático: 54.416 tokens

Al subir la reserva al 98%, 57.344 tokens arrancan y completan correctamente la
prueba larga. Se fija 56k para no usar el techo matemático completo.

## KV FP8 descartada

Con el checkpoint nuevo, KV FP8 + `--calculate-kv-scales` reproduce la misma
corrupción:

```text
月 ... aaaaa...
```

Por tanto, no debe activarse KV FP8 para este modelo con vLLM 0.26.0 en la RTX
3090. El contexto anterior de 112k queda invalidado.

## Corrección de carga en oCabra

La primera carga mediante la ruta pública fue rechazada antes de iniciar vLLM.
El planificador comparaba `gpu_memory_utilization=0.98` con la VRAM disponible
después de descontar también el buffer genérico de oCabra. Ambos mecanismos son
márgenes de seguridad, por lo que se estaban contabilizando dos veces:

- vLLM reserva el 2% de la RTX 3090, unos 492 MB;
- oCabra restaba además su buffer genérico de 512 MB;
- incluso con la GPU vacía, el control rechazaba el perfil por unos 480 MB.

El control de pesos sigue usando la VRAM útil de oCabra, pero el presupuesto de
`gpu_memory_utilization` se compara ahora con la VRAM física libre de NVML. Tras
la corrección, una carga real mediante `/v1/chat/completions` asignó la GPU 1,
inició el worker con contexto 57.344 y devolvió exactamente `OK`.
