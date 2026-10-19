# TesisV4_src

Paquete que clona `multirun.ipynb` en `./src` desde
`emiliodelgadouy/TesisV4_src`. Comentarios y docstrings en español.

```
src/
  notebook.py                 # NotebookRuntime + configure_notebook()
  notebook_api.py             # fachada del notebook (import * y __all__)
  dataset/                    # GCS, etiqueta, splits, tf.data, CLAHE, MixUp
  training/                   # modos, experimento, etapas, umbrales, resume, plan
  tracking/                   # CometTracker, CometRunIndex, CometRunStore
  model_builder/              # grafo Keras por modo
  backbones/                  # extractores registrados por key
```

Llamar a `configure_notebook()` antes de importar `notebook_api`: instala
requirements, importa Comet antes que TensorFlow y aplica mixed precision.

## Modos

- `standard`: entrada nativa del backbone.
- `resized`: un canvas de `CONFIG["RESIZED"]["INPUT_SIZES"]` (enteros o pares
  `(H, W)`). Si la lista no está, el default es 3× el tamaño nativo. El nombre
  puede ser `resized`, `resized_672` o `resized_672x672`; el canvas tiene que
  estar en la lista.
- `patch`, `patch_hardneg`.
- `abmil`: canvas `BAG_GRID × tamaño nativo`, independiente de RESIZED.
- `abmil_patch_hardneg`: transfiere backbone y proyección densa desde
  `patch_hardneg` y crea una salida de bag nueva.

`TrainingMode.parse` acepta mayúsculas, guiones y underscores. `simple` →
`standard` y `full` → `resized`.

Para alinear el preentrenamiento con la grilla ABMIL, entrenar `patch_hardneg`
con `PATCH_HARDNEG.ALIGN_TO_BAG_GRID=True`.

## Contrato del notebook

El driver de `multirun.ipynb` es un plan por fases (`training/plan.py`):
`ExperimentPlan` corre una lista de `RunSpec(mode, backbone, input_size, seed,
pretrained)`, saltea lo terminado y compara contra una referencia por
`val_best_metric` con el margen `PLAN.MARGIN`. `METRIC_TO_MAXIMIZE=pr_auc` y
`USE_CUSTOM_BATCH_SIZE=True`. Los bucles de barrido completo por modo quedan
comentados y también pasan por `plan.run`.

## Resume

Cada corrida guarda su estado, los pesos de la última época y el mejor
checkpoint de cada etapa (`training/resume.py`) en un store:

- `RESUME.BACKEND="comet"` (default, `tracking/comet_store.py`): assets
  `resume_*` del experimento de la corrida, subidos por la REST API en un
  thread. El estado lleva versión (`resume_state_{N}.json`). Un commit sube los
  pesos que faltan, después el estado, y recién entonces borra estados viejos y
  pesos que ya no se referencian.
- `RESUME.BACKEND="dir"`: una carpeta por corrida en `RESUME.DIR`.

Si la sesión se corta, volver a correr el notebook:

- `done`: se saltea. Comet conserva solo los pesos finales (`log_model`); el
  store en carpeta conserva el mejor checkpoint global.
- `training` / `trained`: se saltean las etapas completas, la etapa a medias
  sigue desde la última época (`initial_epoch`) y el log sigue en el mismo
  experimento de Comet (`experiment_key` + `mode="get"`).
- Con `RESUME.SKIP_IF_IN_COMET`, una corrida terminada en Comet desde
  `RESUME.COMET_DONE_SINCE` cuenta como hecha; sus pesos se bajan de Comet si
  hace falta transferirlos.

Se restauran pesos, learning rate y contadores de EarlyStopping y
ReduceLROnPlateau; los momentos de Adam no.

`training/inference.py` reconstruye un modelo entrenado desde un `.weights.h5`
y predice un DataFrame en el orden de sus filas (`predict_with_weights`);
`PairedBootstrap.auc_delta` compara dos modelos sobre las mismas imágenes.

CLAHE (`clip_limit` / `tile_grid`) corre después de lateralizar y antes del
resize o del canvas. El batch de referencia de cada backbone cabe en una A100
80GB (mixed_float16, etapa 3) a tamaño nativo, RESIZED 672 y ABMIL 3×3;
`resolve_batch_size()` lo escala si el canvas o la grilla cambian.
`RESIZED.BATCH_SIZES` y `RESIZED.CACHE`, cuando existen, pisan ese cálculo para
todos los backbones.

Los checkpoints quedan en `checkpoints/` como
`{prefix}_stage_{n}_epoch{NN}.weights.h5`. El prefix es `{modo}_{backbone}`, o
`resized_{H}_{backbone}` si el canvas es cuadrado. Cada etapa conserva el mejor.
Comet recibe la métrica maximizada, la época global del restore y los pesos finales.

Las métricas de `patch` y `patch_hardneg` son de parches asistidos por ROI. El
rendimiento de mamografía completa sale de `standard`, `resized` y los modos ABMIL.

## Estado

El dump de Comet del 2026-09-07 (`simple` / `full`, checkpoint `auc`,
EfficientNet-B0, CheXNet y VGG19) es anterior a este contrato. El estado de
esas corridas y de lo que falta entrenar está en el README del workspace
`TesisV4`.
