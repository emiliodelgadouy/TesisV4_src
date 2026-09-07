"""Modos de entrenamiento disponibles para los experimentos."""

from __future__ import annotations

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
