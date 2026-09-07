from __future__ import annotations

import gc
import random

import numpy as np
import tensorflow as tf


class RandomSeeds:
    """Fija las semillas (random/numpy/tf) desde ``config["GENERAL"]["RANDOM_SEED"]``."""

    @staticmethod
    def set(config) -> None:
        seed = config["GENERAL"]["RANDOM_SEED"]
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


class GpuResources:
    """Libera VRAM/RAM entre experimentos y rompe referencias de builders."""

    @staticmethod
    def release(*, clear_keras_session: bool = True) -> None:
        """Libera VRAM/RAM entre experimentos.

        Usar ``clear_keras_session=False`` solo si un builder activo se reutiliza
        inmediatamente despues (p. ej. patch_hardneg → ABMIL transferido).
        """
        if clear_keras_session:
            tf.keras.backend.clear_session()
        gc.collect()

    @staticmethod
    def dispose_model_builder(builder_obj) -> None:
        """Rompe referencias internas de un builder para que gc libere tras clear_session."""
        if builder_obj is None:
            return
        builder_obj.model = None
        builder_obj.backbone = None
        builder_obj.pretrained_builder = None
