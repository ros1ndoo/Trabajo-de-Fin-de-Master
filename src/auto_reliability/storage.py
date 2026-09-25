"""Publicación atómica de archivos; nunca sobrescribe evidencia original."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def atomic_destination(destination: Path, *, immutable: bool = False) -> Iterator[Path]:
    """Publica un archivo hermano completo; ante fallo elimina solo su temporal. Crear un
    enlace duro es atómico y rechaza reemplazar evidencia existente. Los derivados usan
    reemplazo: los lectores ven la versión anterior o la nueva, no una parcial.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        if immutable:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(destination: Path, payload: Mapping[str, Any], *, immutable: bool = False) -> None:
    """Serializa JSON completamente antes de hacerlo visible a otro proceso."""
    with (atomic_destination(destination, immutable=immutable) as temporary,
          temporary.open("w", encoding="utf-8") as stream):
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
