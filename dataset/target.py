"""Definiciones puras de los objetivos binarios del proyecto."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

TARGET_BIRADS = "birads"
TARGET_MASS = "mass"
LEGACY_TARGET_FULL = "full"
BIRADS_COLUMN = "breast_birads"
BIRADS_NEGATIVE = (1, 2)
BIRADS_POSITIVE = (3, 4, 5)
BIRADS_VALID = (*BIRADS_NEGATIVE, *BIRADS_POSITIVE)
VALID_TARGET_MODES = (TARGET_BIRADS, TARGET_MASS)
_TARGET_ALIASES = {LEGACY_TARGET_FULL: TARGET_BIRADS}


def parse_breast_birads(values: pd.Series) -> pd.Series:
    """Entero BI-RADS 1-5, o NA si el valor no es un assessment valido."""
    digits = values.astype("string").str.extract(r"(\d)", expand=False)
    parsed = pd.to_numeric(digits, errors="coerce")
    return parsed.where(parsed.isin(list(BIRADS_VALID)))


def cls_from_birads(values: pd.Series) -> pd.Series:
    """``0`` si BI-RADS 1-2, ``1`` si 3-5, NA si no hay assessment 1-5."""
    parsed = parse_breast_birads(values)
    out = pd.Series(pd.NA, index=values.index, dtype="Float32")
    out = out.mask(parsed.isin(list(BIRADS_NEGATIVE)), 0.0)
    out = out.mask(parsed.isin(list(BIRADS_POSITIVE)), 1.0)
    return out.astype("float32")


class TargetMode:
    """Objetivo binario: ``birads`` (1-2 vs 3-5) o ``mass`` (solo masas)."""

    BIRADS = TARGET_BIRADS
    MASS = TARGET_MASS
    FULL = LEGACY_TARGET_FULL

    def __init__(self, name: str) -> None:
        self.name = self.parse(name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"TargetMode({self.name!r})"

    @classmethod
    def parse(cls, name: str) -> str:
        target = str(name).strip().lower().replace("-", "_")
        target = _TARGET_ALIASES.get(target, target)
        if target not in VALID_TARGET_MODES:
            raise ValueError(
                f"TARGET_MODE desconocido: {target!r}; usar {VALID_TARGET_MODES} "
                f"(alias legado: {LEGACY_TARGET_FULL!r} -> {TARGET_BIRADS!r})"
            )
        return target

    @classmethod
    def from_config(cls, config: Mapping) -> TargetMode:
        """Resuelve TARGET_MODE y conserva POSITIVE_MODE como alias legacy."""
        general = config.get("GENERAL", config)
        target = general.get("TARGET_MODE")
        legacy = general.get("POSITIVE_MODE")
        if target is None:
            target = legacy
        if target is None:
            raise KeyError("Falta GENERAL.TARGET_MODE (o el alias legacy POSITIVE_MODE)")
        return cls(str(target))

    def masks(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Devuelve mascaras (positiva, negativa) antes de deduplicar imagenes."""
        if self.name == self.MASS:
            return df["Mass"].eq(1), df["cls"].eq(0)
        if self.name == self.BIRADS:
            return df["cls"].eq(1), df["cls"].eq(0)
        raise ValueError(
            f"TARGET_MODE desconocido: {self.name!r}; usar {VALID_TARGET_MODES}"
        )
