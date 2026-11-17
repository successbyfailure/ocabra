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

## Hecho (0.103.0) — paso 2

- **Buscador** dentro del selector: Hugging Face (repos con GGUF; se listan
  solo los ficheros de modelo, sin `mmproj` ni `imatrix`) u Ollama (tags). Al
  añadir uno se crea una receta para llama.cpp (`POST /cloud/recipes`) y queda
  elegido.
  - La clave del modelo es el nombre del fichero GGUF, porque un repo trae
    varias cuantizaciones, o el tag de Ollama.
- **Configurar una receta** antes de lanzar, con la ⚙ de las recetas elegidas
  (`PATCH /cloud/recipes/{id}`): contexto, slots y fijado. Al cambiar contexto
  o slots se quita la reserva de VRAM calculada para los valores viejos.
- **0 data:** confirmado por el usuario. Las recetas de un pod en 0 data se
  guardan igual: solo nombres de modelo y configuración, nada de peticiones.

## Hecho (0.104.0) — backends del nodo

- Al lanzar (y al llevar modelos) se eligen los **backends a instalar**. Van
  primero en la precarga (`kind: "backend"`). Se añaden siempre los que
  necesitan los modelos elegidos. llama.cpp (`llamacpp_mks`) ya viene en la
  imagen y no se instala. Ollama no existe en los nodos.
- **Las sugerencias lo tienen en cuenta:** con backends elegidos, «Recientes»
  y «De este nodo» enseñan los suyos y el resto queda tras «De otros
  backends».
- **La búsqueda también:** solo con llama.cpp busca GGUF. Con otros backends,
  cualquier repo, y de cada repo ofrece las variantes que esos backends pueden
  servir (por `backend_type` y compatibilidades). La receta se crea para ese
  backend (repo entero, registro con ese `backend_type`, como Explorar).

## Pendiente / esquinas

- Recetas de diffusers creadas desde la búsqueda: es el repo entero (lo que
  haría Explorar). Las réplicas de nuestros modelos diffusers, en cambio,
  copian solo los ficheros exactos y el GGUF del transformer. Una receta de
  búsqueda puede ser mucho más grande.
- GGUF partidos en varios ficheros (`-00001-of-0000N`): se ofrece cada parte
  como variante. Hay que comprobar en un pod que la descarga trae todas las
  partes.
- Un pod que muere sin pasar por el hub (fallo del proveedor) solo deja lo
  capturado en los últimos 10 min.
- Dos pods con el mismo modelo y configuraciones distintas: gana la última
  captura (una receta por modelo).
