"""Sinusoidalny embedding czasu t — warunkowanie używane tylko przez dyfuzora
(D-tor). Detektor (S-tor) działa bez niego.

TODO (D1): implementacja embeddingu (margines rozmiaru zgodny z base 32).

Uwaga: raise siedzi W FUNKCJI, nie na poziomie modułu — moduł ma się
importować gładko (testy, __init__), a nie wywalać całego łańcucha importów.
"""

from torch import Tensor


def sinusoidal_embedding(t: Tensor, dim: int = 64) -> Tensor:
    """t: (B,).float [0,1] -> embedding (B, dim). Warunkowanie tylko dyfuzora."""
    raise NotImplementedError("D1: sinusoidalny embedding t")
