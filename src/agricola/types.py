# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Shared public and internal types."""

from enum import Enum


class TestType(Enum):
    SCORE = "score"
    WALD = "wald"


class TraitType(Enum):
    QT = "qt"
    BT = "bt"


__all__ = ["TestType", "TraitType"]
