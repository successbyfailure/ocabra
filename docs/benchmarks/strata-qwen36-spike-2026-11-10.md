# Spike de Strata con Qwen3.6 en RTX 3090 y RTX 3060

Fecha: 2026-11-10

## Decisión

Strata funciona técnicamente como servidor OpenAI-compatible en este host, pero
**no interesa integrarlo todavía como backend de oCabra para Qwen3.6**. El brazo
válido en la RTX 3090 completó streaming, tool calling y cleanup, pero no mejora
el rendimiento ni la capacidad ya disponibles con vLLM. El supuesto brazo
multi-GPU no repartió el modelo: el engine de Qwen3.6 ignoró la opción de split.

Los artefactos del spike se conservan fuera del repositorio para poder repetir
la prueba o evaluar otra familia. No se ha cambiado ningún contrato, backend ni
router de oCabra.

## Configuración

| Elemento | Valor |
|---|---|
| Strata Terminal | 0.2.13.1 |
| Control / build | 0.1.59.1 |
| Engine | 0.1.59, CUDA 13 |
| Modelo | Qwen3.6-35B-A3B UD-Q4_K_XL |
| Contexto | 32.768 tokens |
| Visión | Desactivada |
| Concurrencia | 1 |
| MTP | Activado por el engine (`--mtp 2`) |
| Razonamiento en requests | `reasoning_effort=none` |
| Host | RTX 3060 12 GB + RTX 3090 24 GB, 94,2 GiB RAM |

Se usó `reasoning_effort=none` para medir texto visible y tool calls. Con el
default `high`, una respuesta limitada a 128 tokens puede consumir todo el
presupuesto en `reasoning_content` sin llegar a emitir la llamada forzada.

## Resultado válido: RTX 3090

| Métrica | Resultado |
|---|---:|
| Carga hasta readiness | 83,56 s |
| TTFT, prompt técnico | 187,22 ms |
| Salida, prompt técnico | 75,09 tok/s |
| Tokens de salida | 256 |
| Latencia total | 3.596,64 ms |
| VRAM RTX 3090 cargado | 23.548 MiB |
| VRAM RTX 3060 cargado | 56 MiB |
| RAM reportada por Strata | 72,5 / 94,2 GiB |
| Tool call forzada | `sumar(a=19, b=23)`, correcta |
| VRAM tras stop | 56 MiB / 18 MiB |

El texto técnico fue coherente y la API expuso `/v1/chat/completions`,
`/v1/messages` y `/v1/responses`. La carga utilizó 191 de 256 expertos por capa
en VRAM (13,99 GiB) y mantuvo los 18,75 GiB de expertos en memoria host pinned.
El ciclo de vida terminó sin error ni proceso residual.

La respuesta de dos tokens (`pong`) alcanzó 141,79 tok/s, pero no se usa para
comparar motores porque la muestra es demasiado pequeña.

## Diagnóstico del perfil `split`

El plan de Terminal aceptó `1=main,0=helper` y lanzó el engine con
`--layer-split auto`. Sin embargo, el log del engine contiene:

```text
strata-q36: ignoring --layer-split auto (not an option of this engine)
```

La selección dejó la RTX 3060 como primera GPU visible y el engine ejecutó allí
el grueso del trabajo. La distribución observada fue 11.421 MiB en la 3060 y
614 MiB en la 3090, con solo 20 de 256 expertos por capa en VRAM. El resultado
de 26,75 tok/s y 487,41 ms de TTFT representa ejecución limitada por la 3060,
no layer split. Aunque completó generación, tool call y stop, queda invalidado
como comparación multi-GPU.

Esto descubre además una carencia de validación upstream: Terminal permite
planificar roles que el engine elegido no soporta. Una futura integración debe
comprobar capacidades efectivas del engine y no inferirlas del plan aceptado.

## Comparación con los backends actuales

El baseline histórico de oCabra usa otro checkpoint y otro protocolo, por lo
que sirve para decidir entre opciones desplegables, no como comparación
bit a bit:

| Qwen3.6, salida de hasta 256 tokens | Throughput | Carga | VRAM principal |
|---|---:|---:|---:|
| Strata UD-Q4_K_XL, RTX 3090 | 75,09 tok/s | 83,56 s | 23.548 MiB |
| vLLM GPTQ Int4, RTX 3090 | 120,55 tok/s | 91,55 s | 23.481 MiB |
| Ollama GGUF, opción desplegable histórica | 80,23 tok/s | 23,36 s | no aislada |

Frente a vLLM, Strata fue un 37,7 % más lento en salida, cargó un 8,7 % más
rápido y usó prácticamente la misma VRAM. Frente al resultado histórico de
Ollama quedó un 6,4 % por debajo en throughput. No hay mediciones comparables de
TTFT ni de consumo total de RAM para todos los motores.

## Criterios del spike

| Criterio | Resultado |
|---|---|
| Load, streaming, tools y unload sin intervención | Pasa en 3090 |
| Cleanup de proceso y VRAM | Pasa |
| Texto y tool calling utilizables | Pasa |
| Sin crecimiento sostenido de swap | No instrumentado durante toda la ejecución |
| Ventaja material frente al backend actual | No pasa |
| Split mejora al menos un 10 % o habilita más capacidad | No evaluable; engine sin soporte |

La falta de telemetría continua de swap no cambia el rechazo por rendimiento,
pero debe añadirse antes de aceptar otra familia que dependa más de RAM/SSD.

## Próximo paso recomendado

No implementar `StrataBackend` ni modificar `docs/CONTRACTS.md` en esta fase.
Mantener el harness como prueba reproducible. Si se quiere explorar el valor
diferencial de Strata, abrir un spike separado con Qwen3.8 y estas condiciones:

1. Confirmar antes de descargar que el engine de esa familia soporta layer split.
2. Reservar 66–76 GB adicionales y registrar RAM, swap, I/O y VRAM en serie.
3. Comparar una carga que no pueda resolverse de forma estable en una sola 3090.
4. Repetir cada prompt después de warm-up y reportar mediana, no una sola muestra.
5. Exigir cleanup, cancelación y tool calling además del throughput.

Este follow-up se completó el mismo día con Qwen3.8 Q2_0. Véase
`docs/benchmarks/strata-qwen38-split-spike-2026-11-10.md`: la 3090 sola es
viable, el layer split empeora el rendimiento y el all-cards split no admite
esta pareja heterogénea ni este pack.

## Reproducción

```bash
benchmark/install_strata_spike.sh
python3 benchmark/strata_spike.py preflight
python3 benchmark/strata_spike.py plan --profile 3090
python3 benchmark/strata_spike.py run --profile 3090 --execute
```

Los resultados crudos locales están en:

- `benchmark/results/strata-spike/20261110T050907Z_3090/run.json`
- `benchmark/results/strata-spike/20261110T051110Z_split/run.json`

El segundo fichero se conserva como evidencia del problema de capacidades; no
debe agregarse a tablas como un benchmark multi-GPU válido.
