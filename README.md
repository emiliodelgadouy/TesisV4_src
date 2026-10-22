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
  estar en la lista y ser mayor que el tamaño nativo del backbone (los menores
  o iguales se descartan para ese backbone; pedirlos explícitamente falla).
- `patch`, `patch_hardneg`.
- `abmil`: canvas `BAG_GRID × tamaño nativo`, independiente de RESIZED.
- `abmil_patch_hardneg`: transfiere backbone y proyección densa desde
  `patch_hardneg` y crea una salida de bag nueva.

`TrainingMode.parse` acepta mayúsculas, guiones y underscores. `simple` →
`standard` y `full` → `resized`.

Para alinear el preentrenamiento con la grilla ABMIL, entrenar `patch_hardneg`
con `PATCH_HARDNEG.ALIGN_TO_BAG_GRID=True`.

## Contrato del notebook

El driver de `multirun.ipynb` es una matriz de experimentos (`training/plan.py`):
`ExperimentPlan` corre una lista de `RunSpec(mode, backbone, input_size, seed,
pretrained, label)` y saltea lo terminado. El notebook agrupa las corridas por
pregunta de tesis (`PREGUNTAS`: pregunta → grupo → corridas) y entrena su unión
sin duplicados en el proyecto de Comet `tesis_v4`. El nombre de cada corrida
siempre termina en `_seed{N}` (`resolve_experiment_name`). `METRIC_TO_MAXIMIZE=pr_auc` y
`USE_CUSTOM_BATCH_SIZE=True`. Los bucles de barrido completo por modo quedan
comentados y también pasan por `plan.run`.

- `label`: esquema de `dataset/target.py::LABEL_SCHEMES` (A 1–2 vs 3–5, B sin
  BI-RADS 3, C con 3 negativo). Sufijo `_lbl{x}` en el nombre salvo A, antes
  de `_seed{N}`.
- `predictions(spec, split)`: predice el split canónico completo con los
  pesos finales y lo cachea como asset `predictions_{split}.csv` en Comet.
- `evaluate`, `pairwise`, `seed_summary`: siempre sobre las mismas
  imágenes y la tarea `PLAN.TASK`, con `ClinicalMetrics` y
  `PairedBootstrap` (agrupado por `patient_id`). `level="breast"` agrega por
  mama (`BreastAggregation`: máximo entre vistas por estudio y lateralidad);
  `evaluate(..., ci=True)` suma el IC 95 % por paciente de cada corrida.
- `ClinicalMetrics.compute(y, p, threshold=)`: ROC-AUC, PR-AUC,
  PR-AUC/prevalencia, sensibilidad a especificidad 90/95 %, precisión a recall
  80 %, Brier, ECE y, con umbral, sensibilidad, especificidad, PPV, NPV, F1 y
  exactitud balanceada. El umbral es el Youden de val al mismo nivel.
- Cada corrida loguea en Comet esas métricas para val y test, por imagen
  (`{split}_*`) y por mama (`{split}_breast_*`), la curva `calibration_{split}`
  y, en test, `test_roc_auc_ci_*` / `test_pr_auc_ci_*` (bootstrap por paciente,
  `PLAN.BOOTSTRAP_SAMPLES` muestras).
- `pairwise`: todas contra todas, con diferencia e IC 95 % por par, sin elegir.
- `unique(specs)`: deduplica por nombre conservando el orden; `by_cost` ordena
  de lo más barato a lo más caro.
- `timing(specs)`: horas reales (`training_wall_seconds`) frente al costo
  estimado, y reestimación de lo que falta.
- `status_table(specs, preguntas=)` / `publish_status(...)`: una fila por
  corrida (incluidas las pendientes) con `status`, `status_code`, `progress`,
  horas y métricas. `run_all` la sube antes y después de cada corrida como
  asset `estado_matriz.csv` (se pisa) del experimento `PLAN.STATUS_EXPERIMENT`
  (default `estado_matriz`, tag `estado`), con métricas `runs_*`, `hours_done`
  y `matrix_progress` (avance ponderado por costo).
- `publish_report(table, name)`: sube una tabla del reporte como asset `name`
  (se pisa) del mismo experimento. Nada queda en disco: `log_table_asset`
  escribe un CSV temporal porque `APIExperiment.log_asset` necesita una ruta;
  las predicciones recalculadas suben igual como `predictions_{split}.csv`.
- Cada corrida loguea en su experimento `run_status` (other), `run_status_code`
  y `run_progress` (métricas; el avance es sobre las épocas planificadas de las
  tres etapas) y las coordenadas `matrix_backbone`, `matrix_arch`,
  `matrix_seed`, `matrix_label`. Códigos: −1 `failed`, 0 `pending`, 1
  `partial`, 2 `training`, 3 `evaluating`, 4 `done`.
- Cada corrida loguea `LABEL*`, `EVAL_TASK` y `PROVENANCE_*` (commit de `src`,
  hash de splits y de config, versiones de Python/TF/Keras/comet_ml y GPU); el
  plan falla si hay leakage de paciente.

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
- Con `RESUME.SKIP_IF_IN_COMET`, una corrida terminada en Comet con el mismo
  nombre cuenta como hecha; sus pesos se bajan de Comet si hace falta transferirlos.

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
todos los backbones; con `USE_CUSTOM_BATCH_SIZE`, el batch de la tabla no supera
el escalado del backbone (ViT-S/16 a 672 usa 8).

Los checkpoints quedan en `checkpoints/` como
`{prefix}_stage_{n}_epoch{NN}.weights.h5`. El prefix es `{modo}_{backbone}`, o
`resized_{H}_{backbone}` si el canvas es cuadrado. Cada etapa conserva el mejor.
Comet recibe la métrica maximizada, la época global del restore y los pesos finales.

Las métricas de `patch` y `patch_hardneg` son de parches asistidos por ROI. El
rendimiento de mamografía completa sale de `standard`, `resized` y los modos ABMIL.

## Estado

Las corridas de `tesis_v3` (incluido el dump de Comet del 2026-09-07 con
`simple` / `full` y checkpoint `auc`) son anteriores a la matriz actual. La
matriz y lo que falta entrenar están en el README del workspace
`TesisV4`.
