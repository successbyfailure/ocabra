# Revisión de `REPORT.md` — Benchmark v3

**Para**: el agente que ejecutó el benchmark · **Fecha**: 2026-11-08

Todas las horas son **UTC**, como las de la telemetría del servidor.

---

# 1. Qué está mal

## 1.1 `ravenx` no lee `reasoning_effort`

Son dos modelos distintos y el informe los trata como uno:

| Nombre | Modelo real | ¿Lee los niveles? |
|--------|-------------|-------------------|
| `ravenx` | RavenX-CyberAgent-v6.2 | **No.** Solo tiene interruptor on/off |
| `ravenx-chaos-agent` | RavenX-Chaos-Agent-Qwen3.8-27B | **Sí.** No está en el benchmark |

Verificado extrayendo la plantilla de los GGUF. Los únicos tres modelos del
parque cuya plantilla lee `reasoning_effort` son `qwen3.8:27b`,
`qwen3.8:27b-abliterated` y `ravenx-chaos-agent`. Se probó uno.

## 1.2 Lo medido como "niveles de ravenx" era un bug del servidor

Al no leer los niveles, lo único que variaba era el presupuesto de pensamiento
que inyecta oCabra. Y ese presupuesto se fijaba por nivel **sin mirar el
`max_tokens`**:

```
nivel=medium   max_tokens=4096   -> presupuesto de pensamiento = 16384
nivel=low      max_tokens=512    -> presupuesto = 2048
nivel=high     max_tokens=4096   -> SIN TOPE
```

Con `medium` el modelo podía pensar 16.384 tokens sobre una salida de 4.096:
truncado garantizado. Eso es el "razonamiento infinito, JSON inutilizable" del
informe — no era infinito, era presupuestado por encima de lo que cabía.

Arreglado en **v0.51.2, desplegada a las 02:15 del 08/11**, es decir **después
de que terminase el run (01:58)**. Toda la sección de `reasoning_effort` del
informe se midió contra el bug, y la recomendación "`ravenx`: usar siempre
`off`" no es válida todavía.

## 1.3 El `HTTP 500` de `nemotron3:33b` no existe

La nota ¹ dice "HTTP 500 en M/extract_claims". En la telemetría del servidor el
único fallo de ese modelo en tres días es un **`499 cancelled`** a las
01:08:47: el cliente se rindió tras 120,6 s. No hubo ningún `500`. Un `499` es
"mi timeout era corto" y un `500` es "el servidor está roto"; llevan a
investigar sitios distintos.

## 1.4 Las cifras de `nemotron3:33b` mezclan dos máquinas distintas

Sus 27 peticiones van de 01:00 a 01:19, y dentro de esa ventana cambió de
régimen:

| Ventana | Rendimiento |
|---------|-------------|
| 01:00 → 01:07 | **~9 tok/s** |
| 01:11 → 01:19 | **~100 tok/s** |

Once veces de diferencia en el mismo run. Es el único LLM que queda sobre
Ollama, y Ollama decide el reparto CPU/GPU al cargar. Lo cargó a las 01:02 con
`qwen3:32b` ocupando las dos tarjetas, así que lo repartió **mitad en CPU**:
`ollama ps` reportaba `52%/48% CPU/GPU` y `size_vram` de 1,1 GiB para un modelo
de 27 GB.

Tal como está el informe, **las filas S y M de ese modelo son del régimen CPU y
las de L, calidad y TTFT del régimen GPU**. No son comparables entre sí ni con
el resto de la tabla. (El bug es del servidor y sigue abierto.)

## 1.5 La causa de los tres "FAIL" de contexto largo está mal atribuida

El informe dice "Preset no cubre 47k tokens", que suena a límite artificial de
oCabra. No lo es: el documento de 25.000 palabras son **43.000 tokens reales**,
así que no entra en `qwen3:8b` (32k), `mistral:7b` (32k) ni `qwen3:32b` (40k).
Los FAIL son correctos; el motivo que se da, no.

Medido con el tokenizador de cada modelo:

| Documento | Tokenizador Qwen | Tokenizador Mistral |
|-----------|----------------:|--------------------:|
| 15.000 palabras | 28.908 | 32.024 |
| 25.000 palabras | 46.860 ¹ | 52.133 ¹ |

¹ Incluye la reserva de salida que añade oCabra (4.096 por defecto).

El mismo texto cuesta un **11 % más** con el tokenizador de Mistral. La regla
"palabras × 1,7" no sirve para decidir si algo cabe.

## 1.6 El informe dice cubrir todos los modelos y no es cierto

"Cubre todos los modelos disponibles" y "9 modelos probados, todos
disponibles". Hay **13 perfiles de LLM** y cuatro categorías enteras sin medir
(ver §3).

## 1.7 El encabezado no refleja la versión real del servidor

Dice "Fecha: 2026-11-08 / oCabra: v0.49+". El run fue de **21:48 (07/11) a
01:58 (08/11)** y en esas cuatro horas se desplegaron tres versiones, con una
cuarta justo después. Mezclar cuatro versiones bajo una etiqueta es lo que hace
que un informe envejezca mal.

Un dato concreto de esto: **el resultado de `mistral:7b` a 15.000 palabras
(17 tok/s) existe solo gracias a v0.51.1**, desplegada a las 23:35 del 07/11.
Antes, ese prompt se estimaba en 34.415 tokens contra una ventana de 32.768 y
se rechazaba con `400`; medido de verdad son 32.024, y cabe.

## 1.8 `ravenx` y `gemma4:12b` no completaron en la ventana principal

De las 27 peticiones que caracterizan un run completo, entre las 00:00 y las
02:00 `ravenx` tiene 4 y `gemma4:12b` tiene 1. Sus datos vienen de la parte
anterior del run (desde las 21:49). No invalida nada, pero el informe no dice
de cuándo es cada bloque, y **`gemma4:12b` es el modelo que recomienda para el
caso de uso principal**.

## 1.9 La configuración de Escriba que documenta está obsoleta

- Dice `gemma4:12b (vLLM, ctx 128k)`. **No hay ningún gemma4 en vLLM** en este
  servidor; el único modelo vLLM del parque es el reranker. Funciona porque
  `gemma4:12b` resuelve al despliegue de llama.cpp, pero la nota engaña.
- Su STT es `whisper-large-v3-diarized` cuando el propio benchmark demuestra que
  `whisper-large-v3-turbo` + diarización es mejor y más rápido: WER 4,18 %
  frente a 5,86 %, DER 10,39 % frente a 12,08 %, 17× frente a 15×.

---

# 2. Qué pruebas hay que repetir

## Alta prioridad — cambian conclusiones del informe

**1. Todo `reasoning_effort`, contra v0.51.2 o posterior.**
Lo de la §1.2. Qué cubrir:
- Los **tres** modelos que leen los niveles: `qwen3.8:27b`,
  `qwen3.8:27b-abliterated`, `ravenx-chaos-agent`.
- Los **cuatro** valores: `off`, `low`, `medium`, `high`. `high` no se probó
  nunca, y es el que devolvía `500` hasta v0.50.1 y el que ahora tiene tope por
  primera vez.
- Con `max_tokens` **holgado** (≥ 16.384, para que `medium` no se recorte) y
  también ajustado, para ver el recorte en acción.
- `ravenx` aparte, dejando claro que su único control real es `off` / no-`off`.

Reemplaza las secciones "Modelos con reasoning_effort" y "Configuración de
reasoning_effort" enteras.

**2. `mistral:7b` para salida estructurada.**
"No produjo JSON válido" es uno de los pocos juicios categóricos del informe,
y su tirada fue la última del run (01:19-01:26). Repetir con un prompt dentro
de su ventana (< 25.000 tokens reales) antes de dejarlo escrito.

**3. `nemotron3:33b`: excluirlo o repetirlo con la tarjeta libre.**
Lo de la §1.4. Si se repite, con las dos GPUs libres antes de que Ollama
cargue, y verificando con `ollama ps` que dice `100% GPU` **antes** de medir.
Recomendación: excluirlo y anotar por qué.

## Media prioridad — amplían cobertura

**4. Los cuatro modelos de texto que faltan** (§3.1), al menos velocidad y
calidad de extracción. Dos de ellos entran además en la prueba 1.

**5. Los routers `recomendado-*`** (§3.3): latencia de punta a punta de
`recomendado-sota` y `recomendado-ha-chatbots` frente a nombrar el modelo a
mano, **con la 3090 ocupada por otro modelo grande**. Es el escenario real y es
donde el router debería ganar.

**6. Contexto largo por encima de 43.000 tokens.**
El tramo mayor del informe son 25.000 palabras. `gemma4:26b` admite 262.144
tokens y está verificado a **230.011 tokens** (176 s, sin fallo). Entre 43k y
230k no hay ningún dato, y es donde viven los casos de "resumir un documento
entero".

## Baja prioridad — completitud

**7. TTS, imagen, música y escalado** (§3.2), o quitar la frase "cubre todos los
modelos disponibles".

**8. Segundo pase de lo sensible al arranque en frío.**
Las tablas de concurrencia y TTFT dicen "promedio de 3 runs calientes", pero
varias mediciones de velocidad incluyen la primera petición. En este servidor la
primera paga entre 40 y 60 s de carga para un modelo grande, y **la 3090 solo
tiene sitio para uno**, así que cada cambio de modelo la vuelve a pagar.

---

# 3. Qué falta

## 3.1 Cuatro modelos de texto de trece

| Modelo | Por qué importa |
|--------|-----------------|
| `ravenx-chaos-agent` | Uno de los tres que **sí** leen `reasoning_effort`; destino de `recomendado-sota` |
| `qwen3.8:27b-abliterated` | También lee los niveles; destino de `recomendado-sota` y `recomendado-ha-chatbots` |
| `qwen3-coder:30b` | El único modelo de código del parque, 116k de contexto por defecto |
| `bonsai-27b-ternario` | Backend ternario (bitnet); experimental, pero habilitado |

## 3.2 Cuatro categorías enteras

El alcance declarado (LLM/STT/embeddings/reranker) es legítimo, pero entonces
hay que quitar el "cubre todos los modelos disponibles". Sin medir:

- **TTS**: 4 voces Piper (`piper-es-mx-claude-high`, `glados-es-fine-tuned`,
  `piper-es-es-davefx-medium`, `piper-en-us-lessac-high`). Hay telemetría de
  latencia (~300-500 ms por frase corta) pero nadie ha medido calidad.
- **Imagen**: `qwen-image-2.1` (~114 s por imagen) y `qwen-image-edit-plus`.
- **Música**: `ace-step-turbo`.
- **Escalado**: 5 upscalers de imagen y `flashvsr-v1-1` para vídeo.
- **STT alternativo**: `vibevoice-asr-bitnet`.

## 3.3 Los seis routers `recomendado-*`

Son la vía recomendada para quien no quiere elegir modelo, y nadie ha medido
qué ventaja o qué penalización tienen frente a nombrar el modelo a mano.

## 3.4 El valor `high` de `reasoning_effort`

Solo se probó `off`, `low` y `medium`. El hueco de cobertura coincidía
exactamente con el único valor que estaba roto.

## 3.5 De los tres modelos con niveles de razonamiento, solo se probó uno

Consecuencia de la §1.1, pero vale repetirlo: la sección de razonamiento del
informe describe el parque a partir de un único modelo que además no lee los
niveles.
