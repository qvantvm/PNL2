"""Load and save PNL/2 dataset samples (script + optional reference image)."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

SIDECAR_SUFFIX = ".sample.json"
SAMPLE_VERSION = 1
SAMPLES_DIR_ENV = "PNL2_SAMPLES_DIR"

BLANK_PNL = """pnl/2
score {
    meta {
        title="Untitled"
        profile=[core,notation]
    }
    part piano instrument=piano staves=2 {
        meter at=1:0 beats=4 beat-unit=1/4
        key at=1:0 tonic=C mode=major
        measure 1 {
            staff RH {
                voice RH1 {
                    note n1 pitch=C5 dur=1/4
                }
            }
            staff LH {
                voice LH1 {
                    rest r1 dur=1
                }
            }
        }
    }
}
"""


@dataclass
class Sample:
    """In-memory sample: PNL text plus optional paths for the pair."""

    text: str
    pnl_path: Path | None = None
    expected_path: Path | None = None
    sidecar_path: Path | None = None
    expected_ref: str | None = None

    @property
    def display_name(self) -> str:
        if self.pnl_path is not None:
            return self.pnl_path.name
        return "untitled.pnl"


def new_sample() -> Sample:
    return Sample(text=BLANK_PNL)


def sidecar_path_for(pnl_path: Path) -> Path:
    return pnl_path.with_suffix(SIDECAR_SUFFIX)


def is_sidecar(path: Path) -> bool:
    return path.name.endswith(SIDECAR_SUFFIX)


def load_sample(path: str | Path) -> Sample:
    """Open a ``.pnl`` or ``.sample.json``. Sibling ``.png`` is auto-attached."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sample not found: {path}")
    if is_sidecar(path):
        return _load_sidecar(path)
    if path.suffix.lower() == ".pnl":
        return _load_pnl(path)
    raise ValueError(f"Open a .pnl or .sample.json file, not {path.name}")


def list_samples(directory: str | Path) -> list[Path]:
    """Sorted ``.pnl`` scripts in a dataset folder."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        return []
    return sorted(path.resolve() for path in directory.glob("*.pnl") if path.is_file())


def save_sample(sample: Sample, dest: str | Path | None = None) -> Sample:
    """Write ``.pnl`` and sidecar. In-place saves keep an existing ``expected`` path."""
    pnl_path = Path(dest) if dest is not None else sample.pnl_path
    if pnl_path is None:
        raise ValueError("Save As requires a destination path")
    pnl_path = Path(pnl_path).expanduser()
    if pnl_path.suffix.lower() != ".pnl":
        pnl_path = pnl_path.with_suffix(".pnl")
    pnl_path = pnl_path.resolve()
    pnl_path.parent.mkdir(parents=True, exist_ok=True)
    pnl_path.write_text(sample.text, encoding="utf-8")

    sidecar = sidecar_path_for(pnl_path)
    if _preserve_expected_ref(sample, pnl_path):
        expected_ref = sample.expected_ref
        expected = sample.expected_path
        if expected is None and expected_ref:
            base = sample.sidecar_path.parent if sample.sidecar_path else sidecar.parent
            expected = _resolve_existing(base, expected_ref)
    else:
        expected = _copy_reference(sample.expected_path, pnl_path)
        expected_ref = expected.name if expected is not None else None

    payload = {
        "version": SAMPLE_VERSION,
        "pnl": pnl_path.name,
        "expected": expected_ref,
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return Sample(
        text=sample.text,
        pnl_path=pnl_path,
        expected_path=expected,
        sidecar_path=sidecar,
        expected_ref=expected_ref,
    )


def candidate_sample_dirs() -> list[Path]:
    """Search order for dataset folders (env, harmony extraction, cwd, repo)."""
    dirs: list[Path] = []
    env = os.environ.get(SAMPLES_DIR_ENV)
    if env:
        dirs.append(Path(env).expanduser())
    vibe = _repo_root().parent.parent
    dirs.append(vibe / "music_document_dataset_extraction" / "harmony_dataset" / "samples")
    dirs.append(Path.cwd() / "samples")
    dirs.append(_repo_root() / "samples")
    return dirs


def existing_sample_dirs() -> list[Path]:
    seen: list[Path] = []
    for raw in candidate_sample_dirs():
        try:
            path = raw.expanduser().resolve()
        except OSError:
            continue
        if path.is_dir() and path not in seen:
            seen.append(path)
    return seen


def default_samples_dir() -> Path:
    """Prefer a folder that already has ``.pnl`` samples, else the first existing candidate."""
    existing = existing_sample_dirs()
    for path in existing:
        if list_samples(path):
            return path
    if existing:
        return existing[0]
    return Path.cwd() / "samples"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _expected_ref(data: dict) -> str | None:
    value = data.get("expected")
    return value if isinstance(value, str) and value else None


def _load_pnl(path: Path) -> Sample:
    text = path.read_text(encoding="utf-8")
    sidecar = sidecar_path_for(path)
    expected: Path | None = None
    sidecar_path: Path | None = None
    expected_ref: str | None = None
    if sidecar.is_file():
        sidecar_path = sidecar
        data = _read_sidecar_data(sidecar)
        expected_ref = _expected_ref(data)
        expected = _resolve_existing(sidecar.parent, expected_ref)
    if expected is None:
        expected = _sibling_png(path)
    return Sample(
        text=text,
        pnl_path=path,
        expected_path=expected,
        sidecar_path=sidecar_path,
        expected_ref=expected_ref,
    )


def _load_sidecar(path: Path) -> Sample:
    data = _read_sidecar_data(path)
    pnl = _resolve_existing(path.parent, data.get("pnl"))
    if pnl is None:
        fallback = path.with_name(path.name[: -len(SIDECAR_SUFFIX)] + ".pnl")
        if fallback.is_file():
            pnl = fallback
    if pnl is None or not pnl.is_file():
        raise FileNotFoundError(f"Sidecar {path.name} does not point to a .pnl file")
    text = pnl.read_text(encoding="utf-8")
    expected_ref = _expected_ref(data)
    expected = _resolve_existing(path.parent, expected_ref)
    if expected is None:
        expected = _sibling_png(pnl)
    return Sample(
        text=text,
        pnl_path=pnl,
        expected_path=expected,
        sidecar_path=path,
        expected_ref=expected_ref,
    )


def _preserve_expected_ref(sample: Sample, pnl_path: Path) -> bool:
    """Keep a sidecar image pointer when saving the same ``.pnl`` in place."""
    if not sample.expected_ref or sample.pnl_path is None:
        return False
    if Path(sample.pnl_path).resolve() != pnl_path.resolve():
        return False
    if sample.expected_path is None or sample.sidecar_path is None:
        return True
    resolved = _resolve_existing(sample.sidecar_path.parent, sample.expected_ref)
    if resolved is not None and resolved != Path(sample.expected_path).resolve():
        return False
    return True


def _read_sidecar_data(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid sample sidecar: {path}")
    version = int(data.get("version", SAMPLE_VERSION))
    if version != SAMPLE_VERSION:
        raise ValueError(f"Unsupported sample version {version}")
    return data


def _resolve_existing(base: Path, name: object) -> Path | None:
    if not name or not isinstance(name, str):
        return None
    path = Path(name)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    return path if path.is_file() else None


def _sibling_png(pnl_path: Path) -> Path | None:
    sibling = pnl_path.with_suffix(".png")
    return sibling if sibling.is_file() else None


def _copy_reference(expected: Path | None, pnl_path: Path) -> Path | None:
    if expected is None:
        return None
    expected = Path(expected)
    if not expected.is_file():
        return None
    dest = pnl_path.with_suffix(expected.suffix or ".png")
    if expected.resolve() != dest.resolve():
        shutil.copy2(expected, dest)
    return dest
