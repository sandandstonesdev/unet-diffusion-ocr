"""Inference detekcji (S2): strona -> maska -> lista DetectedLine.

Kontrakt (pełna umowa: docs/contracts.md, D0): splitter NIE skaluje
i NIE normalizuje — surowe wycinki skanu przechodzą do dewarpu.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedLine:
    image: np.ndarray            # (h, w) uint8, surowy wycinek skanu
    bbox: tuple[int, int, int, int]
    baselines: tuple[int, int]   # v0: prosta; margines ~20% x-height
    score: float                 # confidence detekcji (0-1)
    truncated: bool
