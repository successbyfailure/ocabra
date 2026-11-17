# Carga inicial de modelos en los pods

Decidido con el usuario el 2026-11-17, tras perder el nemotron 120B
configurado a mano en un pod al actualizarle la imagen.

## Hecho (0.101.0) — paso 1

- **Recetas** (`core/cloud/recipes.py`, tabla `cloud_model_recipes`, migración
  0047). Una receta es el origen (descarga de HF u Ollama) más la
  configuración del modelo en el pod (contexto, slots, pin…), sin rutas ni
  GPUs de esa máquina.
  - Se capturan preguntando al pod por sus descargas completadas y registradas,
    cada 10 min, antes de terminarlo y antes de cambiarle la imagen.
  - Se omiten los modelos que también existen en el hub: ya salen en «De este
    nodo».
- **Precarga:** cada modelo se puede «cargar al arrancar» o «solo descargar»
  (`download_only`, estado `ready`). El aviso de memoria cuenta solo lo que se
  carga.
  - Antes sumaba el tamaño en disco de todo lo elegido y avisaba de que no
    cabía, aunque solo se quisiera tenerlo descargado. Ese era el «bug» que vio
    el usuario.
  - Al marcar uno que ya no cabe junto a los que se cargan, pasa solo a «solo
    descargar».
- **Selector:** bloques «Recientes en pods» (recetas, con tamaño, contexto,
  slots, pod y cuándo) y «De este nodo». Se puede olvidar una receta.
- **Relanzar** un nodo terminado: misma GPU, número, nube, disco, modo de
  privacidad y modelos, más las recetas capturadas de él (lo puesto a mano).
  `cloud_nodes.privacy_mode` se guarda desde ahora.
- **`update-image`:** vuelve a encolar lo que el pod tenía, también lo puesto a
  mano.

## Pendiente

- **Paso 2 — buscador de Hugging Face / Ollama en el diálogo:**
  - Se reutiliza `/ocabra/registry/hf|ollama` y sus variantes.
  - Elegir repo y cuantización (o tag) crea una receta con configuración por
    defecto: GGUF → `llamacpp_mks`, repo diffusers → imagen.
  - Avisa si no cabe en la GPU elegida.
- **Editar la configuración de una receta** (contexto, slots, pin) antes de
  lanzar. Hoy se usa la que tenía en el pod.
- **Esquinas:**
  - Un pod que muere sin pasar por el hub (fallo del proveedor) solo deja lo
    capturado en los últimos 10 min.
  - Dos pods con el mismo modelo y configuraciones distintas: gana la última
    captura (una receta por modelo).
  - Las recetas de un pod en 0 data se guardan igual: solo nombres de modelo y
    configuración, nada de peticiones (decisión del usuario pendiente de
    confirmar).
