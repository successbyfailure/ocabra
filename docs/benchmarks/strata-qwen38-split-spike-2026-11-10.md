# Spike de Strata Qwen3.8 y split en RTX 3060 + RTX 3090

Fecha: 2026-11-10

## Decisión

Strata sí aporta una capacidad diferencial: sirve **Qwen3.8-Flash-Next Q2_0
completo** en este nodo, con streaming y tool calling correctos. No sustituye a
los backends generales y no debe usar las dos GPU en este host:

- la RTX 3090 sola fue la configuración válida más rápida;
- el layer split 3060 + 3090 fue un 13,8 % más lento;
- el all-cards split engine es incompatible con esta pareja y este pack;
- la carga deja solo unos 4–5 GiB de RAM disponible y provoca presión de swap.

La integración solo se recomienda como backend experimental y opt-in para este
modelo grande: política `on_demand`, ejecución exclusiva respecto a otros LLM,
RTX 3090, un slot, 32K, visión desactivada y DFlash2. No debe promocionarse como
backend general ni configurarse con `pin` o `warm`.

## Qué split ofrece Strata

Strata distingue dos implementaciones:

1. **Layer split:** cada GPU ejecuta un rango contiguo de capas. Cada tarjeta
   mantiene pesos densos, sesión, buffers de prompt y caché de expertos propios.
2. **All-cards split engine:** cada GPU participa en cada capa y mantiene una
   fracción de sus expertos. Tiene requisitos más estrictos de hardware,
   formato, VRAM y RAM.

La documentación upstream advierte que layer split puede añadir capacidad sin
añadir velocidad y que una GPU lenta puede empeorar el decode. El all-cards
split de Strata 0.2.13 exige, entre otras cosas, tarjetas compatibles y todos
los expertos en VRAM. Referencias:

- <https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md>
- <https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md>
- <https://github.com/Niko1221/Strata/releases/tag/v0.2.13>

## Configuración probada

| Elemento | Valor |
|---|---|
| Terminal | Strata 0.2.13.1 |
| Control | 0.1.59.1 |
| Engine | 0.1.59 CUDA 13, sm_86 |
| Modelo | Qwen3.8-Flash-Next Q2_0, ISTA-DASLab |
| Contexto | 32.768 tokens, KV int8 |
| Visión | Desactivada |
| Concurrencia | 1 |
| Repeticiones | 3 por prompt |
| Hardware | RTX 3060 12 GB + RTX 3090 24 GB, 94,2 GiB RAM |

El modelo ocupa 63 GiB en disco. DFlash2 añade 837 MiB, el pack nativo 1,5 GiB
y la capa MTP preparada 6,6 GiB. El prefijo completo del spike ocupa 94 GiB y
quedaron aproximadamente 53 GiB libres en `/docker/ai-models`.

## Resultados

Las cifras de TTFT y throughput son la mediana de tres generaciones del prompt
técnico. La carga de la primera fila incluye la descarga inicial; las siguientes
reutilizan modelo, packs y page cache.

| Modo pedido | Modo efectivo | Drafter | Carga | TTFT | Salida | VRAM 3060 / 3090 | RAM usada |
|---|---|---|---:|---:|---:|---:|---:|
| 3090 | 3090 | DFlash2 | 459,36 s* | 246,41 ms | **69,65 tok/s** | 56 / 23.442 MiB | 90,0 GiB |
| layer split | layer split K=14 | DFlash2 | 78,62 s | 296,92 ms | 60,05 tok/s | 11.279 / 23.596 MiB | 89,7 GiB |
| split engine | fallback layer split | DFlash2 | 91,09 s | 272,81 ms | 66,57 tok/s | 11.279 / 23.596 MiB | 89,8 GiB |
| split engine + MTP | fallback layer split | MTP | 589,10 s** | 270,19 ms | 72,90 tok/s | 11.247 / 23.568 MiB | 90,0 GiB |

\* Incluye 67,6 GB de descarga y preparación de DFlash2; no es carga caliente.

\** Incluye 5,2 GB de descarga y unos ocho minutos de empaquetado local de MTP.
La cifra de rendimiento corresponde al fallback layer split con otro drafter,
no al all-cards split y no se usa para decidir el reparto de GPU.

Todos los brazos completaron tres generaciones cortas, tres técnicas, la tool
call forzada `sumar(a=19, b=23)` y el stop sin proceso ni VRAM residual.

## Layer split efectivo

El log del engine confirma que este brazo sí fue un layer split real:

```text
layer split auto: K=14
layer split: layers 0-13 (CUDA0), 14-47 (CUDA1)
layer split: 80% of the experts resident
```

Las cachés cubrieron aproximadamente el 99,4 % de la masa de routing, pero la
RTX 3060 añadió una etapa lenta y una transferencia por ventana. Respecto a la
3090 sola, el throughput mediano bajó de 69,65 a 60,05 tok/s y el TTFT subió de
246,41 a 296,92 ms. No cumple el umbral del 10 % de mejora; lo empeora.

## Por qué no arrancó el all-cards split

El primer intento con `--split-engine on` y drafting automático cayó a layer
split porque faltaba la capa MTP. Se descargó la capa, se preparó el pack y se
repitió con `--drafting mtp`. Strata volvió a usar el layer split y devolvió los
motivos explícitos:

- RTX 3060 y RTX 3090 no son el mismo modelo y tamaño;
- la embedding Q3_K no se puede convertir a Q8_0 en ese engine;
- el experto compartido Q2_0/Q3_K/Q4_0/Q5_0 no es legible por ese engine;
- no caben todos los expertos: estima unos 41 GiB por tarjeta y solo 11 GiB
  libres en la menor;
- faltaba RAM libre: estimó 42 GiB necesarios y 29 GiB disponibles en ese paso.

No existe por tanto una cifra de all-cards split para este host. Las dos filas
de la tabla que lo solicitan son fallbacks diagnósticos y no benchmarks de esa
implementación.

## Presión de RAM y swap

La 3090 sola dejó entre 3,9 y 5,0 GiB de RAM disponible durante las peticiones.
El engine mantuvo 31,64 GiB de expertos y una tabla PLE de 26,8 GiB en RAM;
redujo automáticamente el chunk de prefill a 8K y desactivó la conversation
cache por falta de memoria.

En la primera carga, `SwapFree` pasó de 47,03 a 44,75 GiB. Durante las
generaciones se mantuvo estable (44,75–44,83 GiB) y subió a 45,51 GiB tras el
stop. Los brazos posteriores repitieron presión adicional antes de recuperar
parte al parar. No hay fuga sostenida durante inferencia, pero sí swapping de
otros procesos/páginas al cargar: el backend debe tratarse como consumidor
exclusivo de RAM, no solo de VRAM.

## Consecuencia para oCabra

El caso diferencial justifica conservar Strata como candidato, pero con un
contrato más estrecho que el planteado inicialmente:

1. Backend experimental, desactivado por defecto y limitado a run configs
   preparados por Strata.
2. Primer perfil: Qwen3.8 Q2_0, GPU1 `main`, GPU0 `off`, 32K, un slot, DFlash2.
3. Reserva de RAM cercana a 90 GiB y precondición de desalojar otros LLM antes
   de cargar; no basta el estimador de VRAM de ModelManager.
4. Política `on_demand`; nunca `pin` o `warm` en este nodo.
5. Rechazar `helper` y `split-engine` para el par 3060 + 3090 mediante una
   capability efectiva, no por aceptación sintáctica del plan.
6. Mantener lifecycle, API key interna, forwarding SSE y cleanup bajo autoridad
   de oCabra como en el plan original.

Antes de producto todavía faltan cancelación en vuelo, una prueba de contexto
largo, convivencia controlada con Whisper en GPU0 y comparación de calidad
contra el Qwen3.8 27B actual. Ninguna de esas pruebas cambia la conclusión de
split: la configuración recomendada en este hardware es la RTX 3090 sola.

## Evidencias locales

- `benchmark/results/strata-spike/20261110T055358Z_qwen38_3090/run.json`
- `benchmark/results/strata-spike/20261110T055555Z_qwen38_split/run.json`
- `benchmark/results/strata-spike/20261110T055808Z_qwen38_split-engine/run.json`
- `benchmark/results/strata-spike/20261110T060853Z_qwen38_split-engine/run.json`

Los dos últimos conservan por separado el fallback sin MTP y la repetición tras
descargar/preparar MTP. Los resultados crudos están ignorados por Git y no se
publican con credenciales.
