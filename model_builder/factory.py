from src.model_builder.abmil import AbmilModelBuilder
from src.model_builder.image import ImageClassifierBuilder
from src.model_builder.resized import ResizedModelBuilder
from src.model_builder.standard import StandardModelBuilder
from src.training.mode import TrainingMode, resolve_abmil_config

_BUILDERS = {
    "standard": StandardModelBuilder,
    "resized": ResizedModelBuilder,
    "patch": ImageClassifierBuilder,
    "patch_hardneg": ImageClassifierBuilder,
    "abmil": AbmilModelBuilder,
    "abmil_patch_hardneg": AbmilModelBuilder,
}


class ModelBuilderFactory:
    """Despacha el builder de Keras segun el modo canonico.

    standard y resized son clases distintas: nativo vs canvas discreto.
    patch / patch_hardneg reusan el grafo de imagen; el crop lo hace el dataset.
    abmil / abmil_patch_hardneg reusan ABMIL; el segundo exige encoder patch.
    """

    @staticmethod
    def class_for(mode) -> type:
        return _BUILDERS[TrainingMode.parse(mode)]

    @staticmethod
    def create(
        config,
        IMG_SIZE,
        backbone,
        preprocess_input,
        *,
        mode="standard",
        initial_bias=None,
        focal_alpha=0.90,
        bag_size=None,
        pretrained_builder=None,
        checkpoint_prefix=None,
        lateralized_inputs=False,
        steps_per_execution=32,
        backbone_trainable=False,
        top_dense=256,
        dropout=0.4,
        learning_rate=1e-3,
        reduce_lr_factor=0.5,
        min_lr=1e-7,
        checkpoint_monitor=None,
        jit_compile=True,
    ):
        # Los hiperparametros compartidos salen del CONFIG del notebook; lo especifico
        # de cada corrida (backbone, mode, initial_bias, ...) sigue llegando por argumento.
        general = config["GENERAL"]
        training = config["TRAINING"]
        abmil_cfg = resolve_abmil_config(config)
        metric_to_maximize = general["METRIC_TO_MAXIMIZE"]
        monitor_mode = "min" if metric_to_maximize == "loss" else "max"

        mode = TrainingMode.parse(mode)
        builder_cls = ModelBuilderFactory.class_for(mode)
        common = dict(
            IMG_SIZE=IMG_SIZE,
            backbone=backbone,
            preprocess_input=preprocess_input,
            backbone_trainable=backbone_trainable,
            top_dense=top_dense,
            dropout=dropout,
            learning_rate=learning_rate,
            focal_alpha=focal_alpha,
            focal_gamma=training["FOCAL_GAMMA"],
            metric_to_maximize=metric_to_maximize,
            checkpoint_monitor=checkpoint_monitor,
            monitor_mode=monitor_mode,
            early_stopping_patience=training["EARLY_STOPPING_PATIENCE"],
            reduce_lr_patience=training["REDUCE_LR_PATIENCE"],
            reduce_lr_factor=reduce_lr_factor,
            min_lr=min_lr,
            aggressive_augmentation=training["AGGRESSIVE_AUGMENTATION"],
            initial_bias=initial_bias,
            pretrained_builder=pretrained_builder,
            jit_compile=jit_compile,
            steps_per_execution=steps_per_execution,
            checkpoint_prefix=checkpoint_prefix,
            lateralized_inputs=lateralized_inputs,
            model_name=mode,
        )
        if TrainingMode.is_mil(mode):
            if mode == TrainingMode.ABMIL_PATCH_HARDNEG and pretrained_builder is None:
                raise ValueError("abmil_patch_hardneg requiere pretrained_builder entrenado en patch_hardneg")
            return builder_cls(
                **common,
                bag_size=bag_size,
                attention_dim=abmil_cfg["ATTENTION_DIM"],
                attention_gated=abmil_cfg["ATTENTION_GATED"],
                bag_grid=abmil_cfg["BAG_GRID"],
                bag_keras_tiling=abmil_cfg["BAG_KERAS_TILING"],
            )
        return builder_cls(**common)


def create_model_builder(*args, **kwargs):
    return ModelBuilderFactory.create(*args, **kwargs)
