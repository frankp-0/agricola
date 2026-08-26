# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Top-level package for agricola."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import TestType, TraitType, step1, step2

__all__ = ["TestType", "TraitType", "step1", "step2"]


def __getattr__(name: str):
    if name in __all__:
        api = import_module(".api", __name__)
        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
