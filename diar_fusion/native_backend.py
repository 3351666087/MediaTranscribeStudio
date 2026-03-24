from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


_BACKEND: Optional["PosteriorFusionNativeBackend"] = None


def _library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    exe_dir = Path(sys.executable).resolve().parent
    if sys.platform == "darwin":
        names = ["libmts_posterior_fusion.dylib"]
    elif os.name == "nt":
        names = ["mts_posterior_fusion.dll", "libmts_posterior_fusion.dll"]
    else:
        names = ["libmts_posterior_fusion.so"]

    paths: list[Path] = []
    for name in names:
        paths.extend(
            [
                root / "build" / "native" / "install" / "lib" / name,
                root / "build" / "native" / "install" / "bin" / name,
                root / "native" / "install" / "lib" / name,
                root / "native" / "install" / "bin" / name,
                exe_dir / "_internal" / "native" / "lib" / name,
                exe_dir / "_internal" / "native" / "bin" / name,
                exe_dir / "native" / "lib" / name,
                exe_dir / "native" / "bin" / name,
            ]
        )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in paths:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


class PosteriorFusionNativeBackend:
    def __init__(self, library: ctypes.CDLL):
        self.library = library
        fn = getattr(library, "mts_viterbi_decode_labels", None)
        if not callable(fn):
            raise RuntimeError("mts_viterbi_decode_labels symbol missing")
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int),
        ]
        fn.restype = ctypes.c_int
        self._decode_labels = fn

    def decode_labels(
        self,
        emissions: np.ndarray,
        boundary_scores: np.ndarray,
        *,
        switch_penalty: float,
        stay_bonus: float,
        boundary_relief: float,
    ) -> Optional[np.ndarray]:
        scores = np.ascontiguousarray(emissions, dtype=np.float32)
        boundary = np.ascontiguousarray(boundary_scores, dtype=np.float32).reshape(-1)
        if scores.ndim != 2 or scores.shape[0] <= 0 or scores.shape[1] <= 0:
            return None
        if boundary.shape[0] != scores.shape[0]:
            return None
        output = np.zeros((scores.shape[0],), dtype=np.int32)
        rc = self._decode_labels(
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            boundary.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(scores.shape[0]),
            int(scores.shape[1]),
            ctypes.c_float(float(switch_penalty)),
            ctypes.c_float(float(stay_bonus)),
            ctypes.c_float(float(boundary_relief)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        if int(rc) != 0:
            return None
        return output


def load_posterior_fusion_backend() -> Optional[PosteriorFusionNativeBackend]:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    for candidate in _library_candidates():
        try:
            if not candidate.exists():
                continue
            _BACKEND = PosteriorFusionNativeBackend(ctypes.CDLL(str(candidate)))
            return _BACKEND
        except Exception:
            continue
    return None
