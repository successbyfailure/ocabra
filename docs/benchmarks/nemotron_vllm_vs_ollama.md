# Benchmark: Nemotron 3 Nano — vLLM AWQ vs Ollama GGUF

Fecha: 2026-07-26

## Objetivo

Comparar el despliegue real de Nemotron 3 Nano en oCabra:

- vLLM 0.19.1: `stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ`.
- Ollama 0.32.0: `nemotron-3-nano:30b`, GGUF Q4_K_M.

La comparación cubre carga con proceso y VRAM inicialmente descargados,
generación individual y lotes de peticiones simultáneas para observar continuous
batching frente al encolado de Ollama.

## Advertencias de comparabilidad

No es una comparación aislada de motores:

- Ambos modelos son Nemotron 3 Nano 31,6B/A3B, pero la cuantización es diferente:
  AWQ (~17 GB en disco) frente a GGUF Q4_K_M (~24,3 GB).
- vLLM usa solo la RTX 3090; Ollama reparte el modelo entre RTX 3090 y RTX 3060.
- Ollama tiene `OLLAMA_NUM_PARALLEL=1`; vLLM tiene `max_num_seqs=12`.
- No se vació la caché ZFS/ARC del host. Las pruebas son cargas frías de
  proceso/VRAM, no lecturas garantizadas desde disco físico.
- No se evaluó calidad de respuesta.

## Configuración

| Parámetro | vLLM | Ollama |
|---|---:|---:|
| Contexto | 65.536 | 65.536 |
| Temperatura | 0 | 0 |
| Semilla | 42 | 42 |
| Secuencias/slots | 12 | 1 |
| Batch tokens | 6.144 | 1 slot |
| Prefix cache | Sí | Sí |
| KV cache | dtype por defecto | q8_0 |
| Ejecución | eager | llama-server |

Para concurrencia se lanzaron peticiones HTTP al mismo tiempo, cada una con un
identificador al principio del prompt y un máximo de 64 tokens. Se midió el
tiempo desde el envío hasta la respuesta completa.

## Carga de proceso y VRAM

| Motor | Prueba 1 | Prueba 2 | Media | Rango |
|---|---:|---:|---:|---:|
| vLLM | 41,08 s | 92,82 s | 66,95 s | 41–93 s |
| Ollama | 66,19 s | 77,52 s | 71,86 s | 66–78 s |

En la primera prueba de Ollama, sus logs desglosan 62,74 s de arranque de
`llama-server`; el tiempo de extremo a extremo fue 66,19 s.

Con solo dos repeticiones y tanta variabilidad, no hay una ventaja concluyente
en carga. vLLM tiene menor media, pero también mucha más dispersión.

### VRAM residente

| Motor | RTX 3060 | RTX 3090 | Total aproximado |
|---|---:|---:|---:|
| vLLM | 43 MiB (base) | 22.581 MiB | 22.581 MiB del modelo/worker |
| Ollama | 9.207 MiB | 15.865 MiB | 25.072 MiB |

Ollama cargó las 53 capas en GPU y creó un único slot de 65.536 tokens.

## Generación individual sostenida

Mismo prompt largo, máximo de 256 tokens:

| Motor | Tokens | Tiempo total | Tokens/s extremo a extremo |
|---|---:|---:|---:|
| vLLM | 256 | 11,35 s | 22,55 |
| Ollama | 256 | 2,67 s | 96,03 |

Ollama reportó 118,41 tokens/s dentro del evaluador, sin contar todo el coste
HTTP y de prefill. En este despliegue, Ollama fue 4,26× más rápido para una
única generación larga. Los `checkpoints` que aparecen en los logs de
`llama-server` pertenecen a su gestión de contexto; no prueban que Ollama esté
aplicando speculative decoding.

## Peticiones simultáneas

### Tiempo total del lote

| Peticiones | vLLM | Ollama | Resultado |
|---:|---:|---:|---|
| 1 | 2,751 s | 1,329 s | Ollama 2,07× más rápido |
| 4 | 2,470 s | 2,859 s | vLLM 1,16× más rápido |
| 8 | 2,392 s | 5,495 s | vLLM 2,30× más rápido |
| 12 | 2,402 s | 8,067 s | vLLM 3,36× más rápido |

### Throughput agregado

| Peticiones | vLLM tokens/s | Ollama tokens/s | vLLM req/s | Ollama req/s |
|---:|---:|---:|---:|---:|
| 1 | 23,26 | 48,15 | 0,363 | 0,752 |
| 4 | 103,65 | 89,53 | 1,619 | 1,399 |
| 8 | 214,02 | 93,18 | 3,344 | 1,456 |
| 12 | 319,78 | 95,21 | 4,997 | 1,488 |

### Latencia por petición

| Peticiones | vLLM media | vLLM máxima | Ollama media | Ollama máxima |
|---:|---:|---:|---:|---:|
| 1 | 2,751 s | 2,751 s | 1,329 s | 1,329 s |
| 4 | 2,469 s | 2,469 s | 1,884 s | 2,858 s |
| 8 | 2,391 s | 2,391 s | 3,237 s | 5,492 s |
| 12 | 2,398 s | 2,401 s | 4,523 s | 8,064 s |

vLLM procesa el lote mediante continuous batching: entre 4 y 12 peticiones el
tiempo total permanece alrededor de 2,4 s. Ollama genera muy rápido en su único
slot, pero el throughput se estabiliza alrededor de 95 tokens/s y la última
petición espera al resto de la cola.

## Conclusión

- **Una petición interactiva:** Ollama es claramente más rápido con la
  configuración actual.
- **Cuatro peticiones:** aparece el punto de cruce; vLLM ya termina antes.
- **Ocho o doce peticiones:** vLLM ofrece mucho más throughput y latencia más
  predecible.
- **Carga:** ambos tardan alrededor de un minuto, con variabilidad alta en vLLM.
- **Uso de hardware:** la ventaja individual de Ollama usa ambas GPUs; vLLM
  conserva la RTX 3060 libre.

Para un asistente-agente con varios consumidores, vLLM es la opción preferible.
Como trabajo posterior conviene probar speculative decoding/MTP en vLLM y
comparar Ollama con más de un slot, midiendo el coste adicional de KV y VRAM.

La actualización y esa prueba están recogidas en
[`nemotron_vllm_upgrade_026.md`](nemotron_vllm_upgrade_026.md).
