# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Numerical preprocessing and linear-algebra helpers."""

from .linear_algebra import assert_covar_full_rank, stdize

__all__ = ["assert_covar_full_rank", "stdize"]
