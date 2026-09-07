# tesis_v3

Paquetes de `src/` usados por `multirun.ipynb`:

```
src/
  notebook.py                 # NotebookRuntime + configure_notebook()
  notebook_api.py             # fachada (mismos nombres que el notebook)
  dataset/                    # I/O, objetivo, splits, tf.data
  training/                   # modos, experimento, etapas, umbrales
  tracking/                   # CometTracker
  model_builder/
  backbones/
```

Modos de entrenamiento:

- `simple`, `full`, `patch`, `patch_hardneg`
- `abmil`
- `abmil_patch_hardneg`: transfiere backbone y proyección densa desde `patch_hardneg`;
  usa una salida de bag nueva

Para la transferencia patch → ABMIL, entrenar `patch_hardneg` con
`PATCH_HARDNEG.ALIGN_TO_BAG_GRID=True`.
