# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

"""Runtime and environment helpers for the agricola CLI."""

from __future__ import annotations

import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from rich.console import Console
from rich.text import Text

logger = logging.getLogger("agricola")
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
console = Console()


def get_version() -> str:
    try:
        return version("agricola")
    except PackageNotFoundError:
        return "unknown"


def print_welcome() -> None:
    art = r"""
                               ░██                      ░██            
                                                        ░██            
 ░██████    ░████████ ░██░████ ░██ ░███████   ░███████  ░██  ░██████   
      ░██  ░██    ░██ ░███     ░██░██    ░██ ░██    ░██ ░██       ░██  
 ░███████  ░██    ░██ ░██      ░██░██        ░██    ░██ ░██  ░███████  
░██   ░██  ░██   ░███ ░██      ░██░██    ░██ ░██    ░██ ░██ ░██   ░██  
 ░█████░██  ░█████░██ ░██      ░██ ░███████   ░███████  ░██  ░█████░██ 
                  ░██                                                  
            ░███████                                                   
                                                                       
"""
    console.print(Text(art, style="bold"))
    console.print(
        f"[bold]agricola[/bold] v{get_version()}\n"
        "[dim]Run --help to see available commands.[/dim]\n"
    )


def setup_logging(log_file: Optional[str], verbose: bool = False) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    log_format = (
        "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s"
        if verbose
        else "%(levelname)s: %(message)s"
    )
    formatter = logging.Formatter(log_format)
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    if log_file:
        if os.path.exists(log_file):
            os.remove(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def report_devices(backend: Optional[str] = None):
    if backend is not None:
        os.environ.setdefault("JAX_PLATFORMS", backend)
    import jax

    devices = jax.devices()
    backend_default = jax.default_backend()
    if backend is not None and backend != backend_default:
        logger.warning(f"backend {backend} not available.\n")
    logger.info(f"Using JAX backend: {backend_default}\n")
    for device in devices:
        logger.info(f"Using device: {device}")
