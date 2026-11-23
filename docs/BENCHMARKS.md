# Benchmarks por aplicación

oCabra puede ejecutar bajo demanda los datasets que mantienen las aplicaciones
que lo consumen. No forman parte de CI, no bloquean despliegues y no aplican
umbrales automáticos: una persona decide qué suites ejecutar y cómo interpretar
la comparación.

Los datos y la lógica de evaluación permanecen en el repositorio de cada
aplicación. oCabra solo contiene el contrato de ejecución, ejecuta el runner
declarado y conserva su resultado sin recalcular la puntuación oficial. La UI
puede derivar una valoración multicriterio para comparar modelos, pero no la
escribe sobre el resultado original.

## Ejecución desde la UI

La página **Benchmarks** (rol `model_manager`) abre en **Resultados** y separa
tres espacios: comparativa gráfica, ejecuciones/histórico y lanzamiento de
pruebas. Agrupa los perfiles por modelo y permite seleccionar sus backends como
burbujas independientes, filtrar el historial, reintentar desde el modelo
fallido y repetir una configuración. Antes de encolar muestra una duración
estimada, su rango y confianza, con desglose por modelo: prioriza muestras
limpias del mismo perfil y configuración, usa la mediana de la suite si no las
hay y escala por número de casos los perfiles de Corrala e IFEval. Una cadena
reanudada suma sus tramos útiles, pero descarta todo lo anterior a una
contaminación tardía; los modelos ya aceptados de una ejecución activa se
incorporan sin esperar a que termine la cola completa. La primera
preparación del dataset y las pausas por tráfico se señalan como tiempo no
incluido. Los resultados conocidos de Escriba,
Corrala, BFCL e IFEval se normalizan para ofrecer rankings por métrica y prueba,
líderes y tabla comparativa; Corrala añade un agregado de calidad ponderado por
casos. El JSON bruto siempre se conserva y puede descargarse junto con el
entorno de cada intento. Un esquema nuevo muestra como alternativa sus métricas
escalares, sin ocultar los datos.

### Ranking global

La portada de Resultados clasifica la combinación exacta `perfil + backend`,
no un nombre de modelo ambiguo. El perfil predeterminado **Producción
equilibrada** reparte 100 puntos así: Corrala 20, Escriba 20, BFCL 20, IFEval
15, fiabilidad 10, rendimiento 10 y eficiencia GPU 5. También hay perfiles de
calidad, latencia e infraestructura eficiente, además de pesos personalizados.

La calidad conserva la escala absoluta que devuelve cada scorer. Para Escriba
se usa la media armónica de cobertura y precisión; Corrala usa su agregado por
casos; BFCL e IFEval usan respectivamente `non_live_accuracy` (o la métrica de
la suite disponible) y `strict_prompt_accuracy`. Los errores observados en las
peticiones reducen la fiabilidad. Rendimiento y eficiencia se normalizan dentro
de cada ejecución, donde todos los participantes compartieron suite, para no
comparar directamente latencias de cargas distintas; la eficiencia divide
además por el número de GPU usadas.

Una batería ausente aporta cero en vez de redistribuir su peso. El modelo debe
tener al menos dos de las cuatro dimensiones de calidad para entrar en la
clasificación; los resultados parciales siguen visibles, pero sin puesto. La UI
muestra cobertura, contribución de cada factor, fuentes y frontera de Pareto
calidad-rendimiento. Los filtros de versión de oCabra y hardware permiten
restringir el cálculo a una cohorte homogénea. Para BFCL se prefiere `core`
sobre `simple_python` y para IFEval `full` sobre `quick`; dentro de la misma
suite se toma el resultado limpio más reciente de cada perfil/backend.

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

Cada intento anota el commit exacto del dataset, las versiones de oCabra,
Python, scorer y módulos de backend disponibles (commit de fuentes en los
nativos, versión de paquete u OCI digest), el perfil/preset/contexto
efectivos, el nodo y todas las tarjetas que sirvieron las llamadas (modelo y
VRAM). También conserva número de peticiones, errores y tokens. Si un runner
falla, **Reintentar modelo** reutiliza su workspace y checkpoint; **Repetir
todo** crea una ejecución nueva. Un intento contaminado nunca entra en el
resultado aceptado: con pausa automática espera tráfico cero y sin ella se
reencola inmediatamente.

### Planes, horarios e histórico

El mismo formulario puede guardarse como plan reutilizable. Un plan fija la
suite, revisión, argumentos y selección exacta de perfiles/backends, y puede
programarse uno o varios días de la semana a una hora en una zona horaria IANA.
La tarjeta del plan muestra próxima y última ejecución, permite lanzarlo ahora,
editarlo, activarlo o eliminarlo. Dos disparos del mismo plan nunca se solapan:
si su ejecución anterior sigue en cola, pausada o corriendo, el vencimiento se
avanza sin duplicarla.

Las ejecuciones terminadas de una suite pueden compararse de cuatro en cuatro.
Una de ellas puede marcarse como referencia única; la tabla histórica muestra
el valor y su delta por modelo y prueba, junto con la versión de oCabra, backend
y GPU registrados. La referencia se devuelve aunque sea más antigua que el
límite normal del historial.

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

**BFCL v4** usa el scorer oficial de Berkeley y la interfaz OpenAI de tools:

- `simple_python`: llamadas simples a funciones Python.
- `core`: todo el bloque no-live (simple Python/Java/JavaScript, múltiple,
  paralela, paralela múltiple e irrelevancia).

**IFEval** usa los prompts y el scorer oficial de Google:

- `quick`: los primeros 50 prompts, útil para iterar.
- `full`: los 541 prompts oficiales.

Las revisiones de BFCL e IFEval están fijadas a un commit, no a una rama
móvil. Sus adaptadores viven en oCabra, guardan una respuesta tras cada caso y
aceptan `--skip-done`; por tanto una pausa continúa por el primer caso pendiente.
La primera ejecución crea entornos aislados con `uv` dentro del cache de
benchmarks. BFCL instala el proyecto oficial completo y ese primer arranque es
sensiblemente más lento; no modifica el entorno Python de la API.
IFEval descarga una sola vez `punkt_tab`, requerido por el tokenizer NLTK de
su scorer oficial, en la misma caché persistente.

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

Los benchmarks públicos cuyo repositorio no ofrece una CLI compatible pueden
declarar `local_runner` en lugar de `script`. El adaptador se incluye en
`backend/ocabra/benchmarks/runners/`, recibe `--source-root` y debe escribir el
resultado dentro del checkout conforme a `result_glob`. Solo se usa para
adaptar transporte y salida: el dataset y el cálculo de la puntuación siguen
siendo los de la revisión oficial fijada.

Las rutas deben ser relativas y no pueden salir del repositorio. El runner usa
`python -I`, pero eso **no es un sandbox**: tanto el manifiesto como la revisión
de la aplicación deben ser de confianza antes de ejecutarlos.

## Alcance de las métricas

El orquestador no recalcula ni corrige puntuaciones. Las definiciones de gold,
normalización, muestreo y métricas pertenecen a cada aplicación y deben
versionarse junto al dataset. Esto permite que Escriba evolucione su evaluación
sin acoplarla al ciclo de releases de oCabra, y que una comparación histórica
pueda volver siempre a la revisión exacta que produjo el resultado.
