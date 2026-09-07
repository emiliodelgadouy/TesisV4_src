"""Tracking de experimentos (Comet ML)."""

__all__ = ["CometEpochLogger", "CometTracker"]


def __getattr__(name: str):
    if name in __all__:
        from src.tracking.comet import CometEpochLogger, CometTracker

        return {"CometEpochLogger": CometEpochLogger, "CometTracker": CometTracker}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
