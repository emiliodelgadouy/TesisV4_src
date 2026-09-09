"""Modos de entrenamiento disponibles para los experimentos."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

TRAINING_MODES = (
    "standard",
    "resized",
    "abmil",
    "abmil_patch_hardneg",
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
    "standard": "standard",
    "simple": "standard",
    "resized": "resized",
    "full": "resized",
    "abmil": "abmil",
    "abmilpatchhardneg": "abmil_patch_hardneg",
    "patch": "patch",
    "patchhardneg": "patch_hardneg",
}

_RESIZED_VARIANT_RE = re.compile(r"^resized(\d+)(?:x(\d+))?$")


class TrainingMode:
    """Modo canonico de entrenamiento. ``parse`` sigue devolviendo el string (p.ej. ``"abmil"``)."""

    STANDARD = "standard"
    RESIZED = "resized"
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
        if cls.parse_resized_size(name) is not None:
            return cls.RESIZED
        key = name.strip().lower().replace("-", "").replace("_", "")
        mode = _NORMALIZED_TO_MODE.get(key)
        if mode is not None:
            return mode
        raise ValueError(
            f"MODE '{name}' no disponible. Opciones: {', '.join(m.upper() for m in TRAINING_MODES)}. "
            "RESIZED admite variantes resized_<H> o resized_<H>x<W> de CONFIG['RESIZED']['INPUT_SIZES']."
        )

    @classmethod
    def parse_resized_size(cls, name: str) -> tuple[int, int] | None:
        """Si el nombre es ``resized_672`` o ``resized_672x448``, devuelve el canvas."""
        key = name.strip().lower().replace("-", "").replace("_", "")
        match = _RESIZED_VARIANT_RE.fullmatch(key)
        if match is None:
            return None
        height = int(match.group(1))
        width = int(match.group(2) or height)
        return (height, width)

    @classmethod
    def is_mil(cls, name: str) -> bool:
        return cls.parse(name) in MIL_MODES

    @classmethod
    def is_patch(cls, name: str) -> bool:
        return cls.parse(name) in PATCH_MODES

    @classmethod
    def is_resized(cls, name: str) -> bool:
        return cls.parse(name) == cls.RESIZED


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


def normalize_input_size(size: int | Sequence[int]) -> tuple[int, int]:
    """Entero → cuadrado; par (H, W) → tupla de ints."""
    if isinstance(size, int):
        value = int(size)
        if value <= 0:
            raise ValueError(f"Tamaño de entrada invalido: {size!r}")
        return (value, value)
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) == 2:
        height, width = int(size[0]), int(size[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Tamaño de entrada invalido: {size!r}")
        return (height, width)
    raise ValueError(f"Tamaño de entrada invalido: {size!r}. Usa un entero o (H, W).")


def resized_size_label(input_size: int | Sequence[int]) -> str:
    """Etiqueta corta para checkpoints/Comet: ``672`` o ``672x448``."""
    height, width = normalize_input_size(input_size)
    if height == width:
        return str(height)
    return f"{height}x{width}"


def _coerce_size_list(raw) -> list[tuple[int, int]]:
    if isinstance(raw, int):
        return [normalize_input_size(raw)]
    if isinstance(raw, tuple):
        return [normalize_input_size(raw)]
    if isinstance(raw, list):
        if not raw:
            raise ValueError("RESIZED.INPUT_SIZES esta vacio.")
        return [normalize_input_size(item) for item in raw]
    raise ValueError(
        f"RESIZED.INPUT_SIZES invalido: {raw!r}. Usa una lista de enteros o pares (H, W)."
    )


def resolve_resized_input_sizes(
    config: Mapping,
    *,
    native_size: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Lista discreta de canvases RESIZED.

    Lee ``CONFIG["RESIZED"]["INPUT_SIZES"]``. Acepta enteros o pares ``(H, W)``.
    ``INPUT_SIZE`` (singular) es un atajo de un solo elemento.
    Si no hay lista, usa 3× el tamaño nativo del backbone.
    """
    resized = dict(config.get("RESIZED") or {})
    raw = resized.get("INPUT_SIZES")
    if raw is None and "INPUT_SIZE" in resized:
        raw = resized["INPUT_SIZE"]
    if raw is None:
        if native_size is None:
            raise ValueError("RESIZED.INPUT_SIZES no esta definido y no hay tamaño nativo para el default 3×.")
        native_h, native_w = normalize_input_size(native_size)
        return [(3 * native_h, 3 * native_w)]
    sizes = _coerce_size_list(raw)
    unique: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for size in sizes:
        if size not in seen:
            unique.append(size)
            seen.add(size)
    return unique


def resolve_resized_input_size(
    config: Mapping,
    *,
    native_size: tuple[int, int] | None = None,
    requested: int | Sequence[int] | None = None,
) -> tuple[int, int]:
    """Elige un canvas de la lista discreta RESIZED."""
    allowed = resolve_resized_input_sizes(config, native_size=native_size)
    if requested is None:
        if len(allowed) == 1:
            return allowed[0]
        pretty = ", ".join(resized_size_label(size) for size in allowed)
        raise ValueError(f"RESIZED define varios tamaños ({pretty}); pasa input_size=... o usa un nombre resized_<H>.")
    size = normalize_input_size(requested)
    if size not in allowed:
        pretty = ", ".join(resized_size_label(item) for item in allowed)
        raise ValueError(f"Tamaño RESIZED {resized_size_label(size)} no esta en la lista discreta: {pretty}.")
    return size


# Geometria para la que estan calibrados ``Backbone.batch_size``.
REFERENCE_RESIZED_SIZE = (672, 672)
REFERENCE_ABMIL_GRID = (3, 3)
_BATCH_SIZE_CAP = 512


def _clamp_explicit_batch(value) -> int:
    batch = int(value)
    if batch <= 0:
        raise ValueError(f"RESIZED.BATCH_SIZES invalido: {value!r}. Usa un entero >= 1.")
    return min(_BATCH_SIZE_CAP, batch)


def _lookup_resized_table(config: Mapping, table_key: str, input_size: int | Sequence[int], *, coerce):
    """Lee ``RESIZED[table_key]`` para un canvas.

    Dict ``{448: valor}`` o lista paralela a ``INPUT_SIZES``. ``None`` si no hay tabla.
    """
    raw = dict(config.get("RESIZED") or {}).get(table_key)
    if raw is None:
        return None
    target = normalize_input_size(input_size)
    if isinstance(raw, Mapping):
        table = {normalize_input_size(key): coerce(value) for key, value in raw.items()}
        if target not in table:
            pretty = ", ".join(resized_size_label(key) for key in table)
            raise ValueError(
                f"RESIZED.{table_key} no define valor para {resized_size_label(target)}. Claves: {pretty}."
            )
        return table[target]
    if isinstance(raw, list):
        sizes = resolve_resized_input_sizes(config)
        if len(raw) != len(sizes):
            raise ValueError(
                f"RESIZED.{table_key} tiene {len(raw)} valores y INPUT_SIZES {len(sizes)}."
            )
        for size, value in zip(sizes, raw):
            if size == target:
                return coerce(value)
        raise ValueError(f"RESIZED.{table_key} no define valor para {resized_size_label(target)}.")
    raise ValueError(
        f"RESIZED.{table_key} invalido: {raw!r}. Usa un dict {{tamaño: valor}} o una lista paralela a INPUT_SIZES."
    )


def resolve_resized_batch_size(
    config: Mapping,
    input_size: int | Sequence[int],
) -> int | None:
    """Batch de ``RESIZED.BATCH_SIZES`` para este canvas, o None si no hay tabla.

    Si la tabla existe, el canvas tiene que estar; si no, se sigue el batch del
    backbone escalado por area.
    """
    return _lookup_resized_table(config, "BATCH_SIZES", input_size, coerce=_clamp_explicit_batch)


def resolve_resized_cache(
    config: Mapping,
    input_size: int | Sequence[int],
) -> bool | None:
    """Cache de ``RESIZED.CACHE`` para este canvas, o None si no hay tabla."""
    return _lookup_resized_table(config, "CACHE", input_size, coerce=bool)


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
    """Baja (o sube) el batch si RESIZED/ABMIL no son la geometria de referencia.

    RESIZED escala con el area del canvas (cuadrado del area si ``memory_scale``
    es ``attention``, p.ej. ViT). ABMIL escala con G²; los tiles siguen a S
    nativo, asi que ViT no paga atencion cuadratica del canvas.
    """
    mode = TrainingMode.parse(mode)
    scaled = float(base)
    if mode == TrainingMode.RESIZED and input_size is not None:
        height, width = int(input_size[0]), int(input_size[1])
        area = max(height * width, 1)
        ref_h, ref_w = REFERENCE_RESIZED_SIZE
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

    Despues escala por ``RESIZED.INPUT_SIZES`` / ``ABMIL.BAG_GRID`` para no OOM,
    salvo que ``RESIZED.BATCH_SIZES`` fije el batch del canvas (mismo valor para
    todos los backbones).
    Devuelve ``(batch_size, source, batch_size_base)``.
    """
    mode = TrainingMode.parse(mode)
    if mode == TrainingMode.RESIZED and input_size is not None:
        explicit = resolve_resized_batch_size(config, input_size)
        if explicit is not None:
            return explicit, "resized", explicit
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
