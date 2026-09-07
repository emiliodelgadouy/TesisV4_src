"""Definiciones puras de los objetivos binarios del proyecto."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

TARGET_MASS = "mass"
LEGACY_TARGET_FULL = "full"
VALID_TARGET_MODES = (TARGET_MASS, LEGACY_TARGET_FULL)


class TargetMode:
    """Objetivo binario: ``mass`` (solo masas) o ``full`` (cualquier hallazgo)."""

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
        if target not in VALID_TARGET_MODES:
            raise ValueError(
                f"TARGET_MODE desconocido: {target!r}; usar {VALID_TARGET_MODES}"
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
        if self.name == self.FULL:
            return df["cls"].eq(1), df["cls"].eq(0)
        raise ValueError(
            f"TARGET_MODE desconocido: {self.name!r}; usar {VALID_TARGET_MODES}"
        )
