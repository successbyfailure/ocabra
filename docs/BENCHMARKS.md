# Benchmarks manuales por aplicación

oCabra puede ejecutar bajo demanda los datasets que mantienen las aplicaciones
que lo consumen. No forman parte de CI, no bloquean despliegues y no aplican
umbrales automáticos: una persona decide qué suites ejecutar y cómo interpretar
la comparación.

Los datos y la lógica de evaluación permanecen en el repositorio de cada
aplicación. oCabra solo contiene adaptadores en `benchmark/apps/`, ejecuta el
runner declarado y archiva el resultado bruto junto con el contexto de la
medición.

## Suites disponibles

```bash
python3 -m benchmark.app_benchmarks list
```

El primer adaptador es **Escriba** y expone cuatro suites:

- `llm`: extracción de claims y velocidad sobre transcripciones de reuniones.
- `stt`: WER y velocidad de transcripción sobre VoxPopuli en español.
- `stt_diarization`: DER, JER, hablantes detectados y velocidad sobre cuatro
  reuniones AMI SDM multihablante con anotación temporal gold.
- `embeddings`: recuperación, clustering y reranking sobre evidencia de reuniones.

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
