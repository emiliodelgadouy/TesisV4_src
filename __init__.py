"""Helpers for CV training notebooks."""

__all__ = [
    "BACKBONES",
    "get_backbone",
    "resolve_backbone",
]


def __getattr__(name):
    """Carga TensorFlow/backbones solo cuando el caller realmente los solicita."""
    if name in __all__:
        from .backbones import BACKBONES, get_backbone, resolve_backbone

        return {
            "BACKBONES": BACKBONES,
            "get_backbone": get_backbone,
            "resolve_backbone": resolve_backbone,
        }[name]
    raise AttributeError(name)
