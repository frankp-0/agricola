# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Formatting helpers for CLI output and option diagnostics."""

from __future__ import annotations


def list_from_csv(arg: str | None) -> list[str] | None:
    return None if arg is None else [x.strip() for x in arg.split(",")]


def get_options_msg(options: dict[str, str]) -> str:
    option_msg = ["Command options:"]
    for key, value in options.items():
        option_msg.append(f"  {key} = {value}")
    return "\n".join(option_msg) + "\n"
