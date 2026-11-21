"""Degradacje: G (dyfuzyjna, q_sample) oraz R (realistyczne zakłócenia).

TODO (D0):
    - G: q_sample z configs/schedule.py na siatce t_degr (200 zdegradowanych
      próbek — definicja gotowości D0)
    - R: degradacje realistyczne jako alternatywa/uzupełnienie G
"""

from typing import Literal

from torch import Tensor

# Rodzaje degradacji R — kontrakt przed implementacją (D0): wartości literalne,
# żeby sygnatura degrade_r była typu przed napisaniem ciała.
RKind = Literal[
    "blur_gauss", "blur_motion", "noise_gauss", "salt_pepper",
    "bleed_through", "jpeg", "contrast",
]


def degrade_g(x0_clean: Tensor, t_frac: float, schedule, seed=None) -> Tensor:
    """G: czysta linia -> q_sample(x0, t).

    Zwraca (1,32,W) [0,1] (clamp po powrocie z [-1,1]).
    """

def degrade_r(x0_clean: Tensor, kind: RKind, seed=None) -> Tensor:
    """R: gauss/motion blur, szum gaussowski, salt&pepper, bleed-through
    (odbicie lustrzane + alpha), JPEG artifacts, kontrast/jasność.
    Operacje na (1,32,W) [0,1]; W niezmienna; tło ≠ atramentu nie rozlewa się."""
