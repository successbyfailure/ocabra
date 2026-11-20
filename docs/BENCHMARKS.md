# Benchmarks por aplicación

oCabra puede ejecutar bajo demanda los datasets que mantienen las aplicaciones
que lo consumen. No forman parte de CI, no bloquean despliegues y no aplican
umbrales automáticos: una persona decide qué suites ejecutar y cómo interpretar
la comparación.

Los datos y la lógica de evaluación permanecen en el repositorio de cada
aplicación. oCabra solo contiene el contrato de ejecución, ejecuta el runner
declarado y conserva su resultado sin recalcular la puntuación.

## Ejecución desde la UI

La página **Benchmarks** (rol `model_manager`) permite buscar y seleccionar
perfiles de ejecución, filtrar el historial, repetir una configuración y
consultar resultados, intentos y logs. Los resultados conocidos de Escriba y
Corrala se normalizan solo para presentarlos en una tabla comparativa; el JSON
bruto siempre se conserva y puede descargarse. Un resultado con un esquema
nuevo muestra como alternativa sus métricas escalares, sin ocultar los datos.

La unidad de trabajo es `aplicación + suite + modelo`: los modelos se ejecutan
secuencialmente para que una contaminación afecte solo al modelo en curso. La
barra de progreso cuenta únicamente modelos aceptados; el historial de
intentos explica cuáles se descartaron y si coincidieron con tráfico externo.

El ejecutor crea una identidad y una API key internas efímeras. Mientras corre
observa las peticiones en vuelo de `ModelManager`; cualquier petición de otra
identidad se considera tráfico real. Con la pausa automática activa:

1. termina el subprocess del runner (no congela una inferencia a medias);
2. marca el intento como interrumpido o contaminado;
3. conserva checkout, resultados parciales y log en `/data/benchmarks`;
4. espera `BENCHMARK_QUIET_PERIOD_S` sin tráfico;
5. vuelve a invocar el runner con `--skip-done` o, si ya había terminado
   contaminado, desde un checkout nuevo.

`request_stats` se consulta al terminar como segunda red: detecta peticiones
cortas que hayan empezado y acabado entre dos sondeos. Solo un intento sin
solapes entra en los resultados limpios. Una pausa manual usa el mismo
checkpoint, pero nunca se reanuda sola.

Corrala requiere `uv` (incluido en la imagen API) y una PostgreSQL desechable.
Por defecto oCabra crea `<base principal>_bench`; se puede indicar otra con
`BENCHMARK_CORRALA_DATABASE_URL`, cuyo nombre debe contener
`bench`/`test`/`scratch`/`tmp`. Para repositorios privados se configura un token
de solo lectura en `BENCHMARK_GIT_TOKEN`; se pasa a Git por entorno y no se
escribe en comandos ni logs.

## Suites disponibles

```bash
python3 -m benchmark.app_benchmarks list
```

**Escriba** expone cuatro suites:

- `llm`: extracción de claims y velocidad sobre transcripciones de reuniones.
- `stt`: WER y velocidad de transcripción sobre VoxPopuli en español.
- `stt_diarization`: DER, JER, hablantes detectados y velocidad sobre cuatro
  reuniones AMI SDM multihablante con anotación temporal gold.
- `embeddings`: recuperación, clustering y reranking sobre evidencia de reuniones.

**Corrala** expone `llm`: 83 casos de triage Zammad, routing de GLaDOS,
subagentes y bucles, con perfiles `quick` (34), `standard` (54), `full` (83)
y `final` (83 × 3 pasadas).

## Ejecutar Escriba

La URL nunca tiene un valor por defecto para evitar medir accidentalmente el
servidor de producción. La clave se lee del entorno: el orquestador no la añade
al comando ni a los metadatos. Como `runner.log` conserva la salida de un script
externo, solo deben ejecutarse revisiones de confianza que no impriman secretos.

```bash
export OPENAI_COMPATIBLE_URL=https://ocabra.example.org
export OPENAI_COMPATIBLE_KEY=...

# Clona Escriba y fija exactamente esta revisión.
python3 -m benchmark.app_benchmarks run escriba llm \
  --revision c2fa051095f11339111dcd9d1aacef337f3e0ee2 \
  --models gemma4:12b,gemma4:26b

# Usa un checkout local; queda anotado si contiene cambios sin commit.
python3 -m benchmark.app_benchmarks run escriba embeddings \
  --source /ruta/a/escriba \
  --models qwen3-embedding:0.6b,qwen3-embedding:4b
```

STT necesita descargar los WAV la primera vez. `--prepare` ejecuta el script
de preparación declarado por Escriba antes del benchmark:

```bash
python3 -m benchmark.app_benchmarks run escriba stt \
  --source /ruta/a/escriba \
  --prepare \
  --models whisper-large-v3-turbo
```

La suite de diarización descarga una selección fija de AMI SDM cuya revisión
y reuniones mantiene Escriba. Un smoke de una reunión se lanza así:

```bash
python3 -m benchmark.app_benchmarks run escriba stt_diarization \
  --source /ruta/a/escriba \
  --prepare \
  --models whisper-large-v3-diarized \
  --suite-arg=--limit \
  --suite-arg=1
```

Para pasar opciones propias del runner se repite `--suite-arg`. Si el valor
empieza por `--`, se usa la forma con signo igual para que `argparse` no lo
confunda con una opción del orquestador:

```bash
python3 -m benchmark.app_benchmarks run escriba llm \
  --source /ruta/a/escriba \
  --models qwen3.8:27b \
  --suite-arg=--reasoning \
  --suite-arg=--skip-done
```

Antes de una ejecución larga se puede inspeccionar la revisión y el comando:

```bash
python3 -m benchmark.app_benchmarks run escriba llm \
  --source /ruta/a/escriba \
  --base-url http://localhost:8000 \
  --dry-run
```

## Resultado archivado

Cada ejecución crea un directorio ignorado por Git:

```text
benchmark/results/<aplicación>/<suite>/<fecha>_<aplicación>_<suite>_<revisión>/
├── run.json       # contexto reproducible, sin credenciales
├── results.json   # salida bruta producida por la aplicación
├── runner.log
└── prepare.log    # solo cuando se usó --prepare
```

`run.json` conserva:

- revisión y estado limpio/sucio del repositorio de la aplicación;
- hash del contenido versionable del dataset;
- versión que devuelve `GET /version` y URL medida;
- modelos, argumentos, tiempos, Python y plataforma;
- revisión local de oCabra desde la que se lanzó el runner.

Los audios descargados no entran en el hash para no leer cientos de megabytes
en cada ejecución. Si se descargan de forma determinista, su manifiesto, rutas
de origen y script de descarga sí quedan incluidos.

## Añadir otra aplicación

Se crea un JSON en `benchmark/apps/<aplicación>.json` con:

- repositorio y revisión predeterminada;
- raíz del dataset dentro de ese repositorio;
- por cada suite: script, patrón del resultado y opción de modelos;
- opcionalmente, un script de preparación de assets.

Las rutas deben ser relativas y no pueden salir del repositorio. El runner usa
`python -I`, pero eso **no es un sandbox**: tanto el manifiesto como la revisión
de la aplicación deben ser de confianza antes de ejecutarlos.

## Alcance de las métricas

El orquestador no recalcula ni corrige puntuaciones. Las definiciones de gold,
normalización, muestreo y métricas pertenecen a cada aplicación y deben
versionarse junto al dataset. Esto permite que Escriba evolucione su evaluación
sin acoplarla al ciclo de releases de oCabra, y que una comparación histórica
pueda volver siempre a la revisión exacta que produjo el resultado.
