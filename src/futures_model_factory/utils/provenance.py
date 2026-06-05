from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import uuid


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    config_path: str
    data_snapshot_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    """Hash a file for lightweight config/data provenance."""
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run_provenance(config_path: str | Path, data_paths: list[str | Path]) -> RunProvenance:
    """Create a run ID and data snapshot ID from immutable input paths."""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in data_paths):
        digest.update(str(path).encode())
        digest.update(file_sha256(path).encode())
    return RunProvenance(
        run_id=str(uuid.uuid4()),
        config_path=str(config_path),
        data_snapshot_id=digest.hexdigest(),
    )


def write_run_provenance(provenance: RunProvenance, path: str | Path) -> None:
    """Write provenance as JSON."""
    Path(path).write_text(json.dumps(provenance.to_dict(), indent=2, sort_keys=True))
