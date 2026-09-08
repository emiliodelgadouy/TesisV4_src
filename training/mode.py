"""Modos de entrenamiento disponibles para los experimentos."""

from __future__ import annotations

from collections.abc import Mapping

TRAINING_MODES = (
    "simple",
    "abmil",
    "abmil_patch_hardneg",
    "full",
    "patch",
    "patch_hardneg",
)

MIL_MODES = (
    "abmil",
    "abmil_patch_hardneg",
)

PATCH_MODES = (
    "patch",
    "patch_hardneg",
)

# Clave sin guiones ni underscores (ver TrainingMode.parse).
_NORMALIZED_TO_MODE = {
    "simple": "simple",
    "abmil": "abmil",
    "abmilpatchhardneg": "abmil_patch_hardneg",
    "full": "full",
    "patch": "patch",
    "patchhardneg": "patch_hardneg",
}


class TrainingMode:
    """Modo canonico de entrenamiento. ``parse`` sigue devolviendo el string (p.ej. ``"abmil"``)."""

    SIMPLE = "simple"
    FULL = "full"
    PATCH = "patch"
    PATCH_HARDNEG = "patch_hardneg"
    ABMIL = "abmil"
    ABMIL_PATCH_HARDNEG = "abmil_patch_hardneg"

    ALL = TRAINING_MODES
    MIL = MIL_MODES
    PATCH_SET = PATCH_MODES

    def __init__(self, name: str) -> None:
        self.name = self.parse(name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"TrainingMode({self.name!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrainingMode):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == self.parse(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    @classmethod
    def parse(cls, name: str) -> str:
        key = name.strip().lower().replace("-", "").replace("_", "")
        mode = _NORMALIZED_TO_MODE.get(key)
        if mode is not None:
            return mode
        raise ValueError(
            f"MODE '{name}' no disponible. Opciones: {', '.join(m.upper() for m in TRAINING_MODES)}"
        )

    @classmethod
    def is_mil(cls, name: str) -> bool:
        return cls.parse(name) in MIL_MODES

    @classmethod
    def is_patch(cls, name: str) -> bool:
        return cls.parse(name) in PATCH_MODES


def resolve_abmil_config(config: Mapping) -> dict:
    """Config de ABMIL: ``CONFIG["ABMIL"]``.

    Acepta el alias legado ``MIL`` y ``FULL["BAG_GRID" / "BAG_CANVAS_MODE"]``.
    Si hay ``ABMIL``, pisa las claves del legado.
    """
    merged: dict = {}
    merged.update(config.get("MIL") or {})
    full = config.get("FULL") or {}
    if "BAG_GRID" in full:
        merged.setdefault("BAG_GRID", full["BAG_GRID"])
    if "BAG_CANVAS_MODE" in full:
        merged.setdefault("BAG_CANVAS_MODE", full["BAG_CANVAS_MODE"])
    merged.update(config.get("ABMIL") or {})
    merged.setdefault("BAG_GRID", (3, 3))
    merged.setdefault("BAG_CANVAS_MODE", "resize")
    merged.setdefault("BAG_KERAS_TILING", True)
    merged.setdefault("ATTENTION_DIM", 128)
    merged.setdefault("ATTENTION_GATED", True)
    return merged


# Geometria para la que estan calibrados ``Backbone.batch_size``.
REFERENCE_FULL_SIZE = (672, 672)
REFERENCE_ABMIL_GRID = (3, 3)
_BATCH_SIZE_CAP = 512


def _floor_power_of_two(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def scale_batch_size(
    base: int,
    mode: str,
    *,
    input_size: tuple[int, int] | None = None,
    bag_grid: tuple[int, int] | None = None,
    memory_scale: str = "spatial",
) -> int:
    """Baja (o sube) el batch si FULL/ABMIL no son la geometria de referencia.

    FULL escala con el area del canvas (cuadrado del area si ``memory_scale``
    es ``attention``, p.ej. ViT). ABMIL escala con G²; los tiles siguen a S
    nativo, asi que ViT no paga atencion cuadratica del canvas.
    """
    mode = TrainingMode.parse(mode)
    scaled = float(base)
    if mode == TrainingMode.FULL and input_size is not None:
        height, width = int(input_size[0]), int(input_size[1])
        area = max(height * width, 1)
        ref_h, ref_w = REFERENCE_FULL_SIZE
        ratio = (ref_h * ref_w) / area
        if memory_scale == "attention":
            ratio *= ratio
        scaled = base * ratio
    elif mode in MIL_MODES and bag_grid is not None:
        instances = max(int(bag_grid[0]) * int(bag_grid[1]), 1)
        ref_instances = REFERENCE_ABMIL_GRID[0] * REFERENCE_ABMIL_GRID[1]
        scaled = base * ref_instances / instances
    return _floor_power_of_two(min(_BATCH_SIZE_CAP, max(1, int(scaled))))


def resolve_batch_size(
    config: Mapping,
    mode: str,
    backbone_name: str,
    *,
    input_size: tuple[int, int] | None = None,
    bag_grid: tuple[int, int] | None = None,
) -> tuple[int, str, int]:
    """Batch de la corrida: custom por backbone×modo, o el global GENERAL/ABMIL.

    Lo decide ``GENERAL["USE_CUSTOM_BATCH_SIZE"]``:

    - ``True``: ``Backbone.batch_size[mode]`` del provider; si falta, cae al global.
    - ``False`` (default): ``ABMIL["BATCH_SIZE"]`` en modos MIL, si no ``GENERAL["BATCH_SIZE"]``.

    Despues escala por ``FULL.INPUT_SIZE`` / ``ABMIL.BAG_GRID`` para no OOM.
    Devuelve ``(batch_size, source, batch_size_base)``.
    """
    mode = TrainingMode.parse(mode)
    general = config["GENERAL"]
    memory_scale = "spatial"
    try:
        from src.backbones import get_backbone

        backbone = get_backbone(backbone_name)
        memory_scale = getattr(backbone, "batch_memory_scale", "spatial")
    except ValueError:
        backbone = None

    if general.get("USE_CUSTOM_BATCH_SIZE") and backbone is not None:
        custom = backbone.batch_size_for(mode)
        if custom is not None:
            base = custom
            source = "backbone"
        elif TrainingMode.is_mil(mode):
            abmil = resolve_abmil_config(config)
            base = int(abmil.get("BATCH_SIZE", general["BATCH_SIZE"]))
            source = "abmil" if "BATCH_SIZE" in abmil else "general"
        else:
            base = int(general["BATCH_SIZE"])
            source = "general"
    elif TrainingMode.is_mil(mode):
        abmil = resolve_abmil_config(config)
        if "BATCH_SIZE" in abmil:
            base = int(abmil["BATCH_SIZE"])
            source = "abmil"
        else:
            base = int(general["BATCH_SIZE"])
            source = "general"
    else:
        base = int(general["BATCH_SIZE"])
        source = "general"

    if bag_grid is None:
        bag_grid = resolve_abmil_config(config).get("BAG_GRID")
    scaled = scale_batch_size(
        base,
        mode,
        input_size=input_size,
        bag_grid=bag_grid,
        memory_scale=memory_scale,
    )
    return scaled, source, int(base)
