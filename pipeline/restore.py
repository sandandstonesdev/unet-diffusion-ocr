"""Restauracja SDEdit (DDIM od t0 do 0) + brama/skip.

Kontrakt: przelotowy — ta sama W, ta sama geometria baseline'u.
Wewnętrznie pracuje w [-1,1]; konwersje zamknięte w środku.
Padding zerami z prawej robi dopiero collate CRNN — restaurator
go nie dotyka.

TODO (D2):
    - próbkowanie DDIM od t0 (siatka {0.2, 0.3, 0.5, 0.7})
    - brama decydująca restore/skip
"""

from torch import Tensor


def restore(degraded: Tensor, t0: float | None = None) -> Tensor:
    """in: (1, 32, W) [0,1]  ->  out: (1, 32, W) [0,1]."""
    raise NotImplementedError("D2: pętla SDEdit")
