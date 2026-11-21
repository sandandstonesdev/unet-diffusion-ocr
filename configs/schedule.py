"""Harmonogram dyfuzyjny: cosinusowy + q_sample (G-degradacja).

Używa go WYŁĄCZNIE dyfuzor (D-tor) — detektor nie ma warunkowania czasem.
Kontrakty (gwarancje ᾱ/β, tolerycje q_sample): docs/contracts.md, sekcje 2-3.
Testy: tests/test_schedule.py (port asercji z notebooks/d0_schedule_inspect).

Intuicja: obrazek w kroku t to KAWAŁEK ORYGINAŁU + KAWAŁEK SZUMU.

- ᾱ_t (alphas_cumprod) — "ile % linii przetrwało": 1.0 = czysta linia, ~0 = szum,
- 1 - ᾱ_t — "ile % czystego szumu doszło",
- pierwiastki przy obu składnikach trzymają skalę: wariancje sumują się do 1,
  więc obrazek nie robi się po drodze ani ciemniejszy, ani jaśniejszy.

Harmonogram jest COSINUSOWY (Nichol & Dhariwal), nie liniowy: liniowy zabija
sygnał ok. 50 kroków przed końcem (koniec marnuje się na jazdę po szumie),
kosinus schodzi do ~0 dopiero przy t = T.
"""

import numpy as np
import torch


def cosine_schedule(T: int = 300, s: float = 0.008, beta_max: float = 0.999):
    """Tablice harmonogramu: betas (dł. T) + alphas_cumprod (dł. T).

    ᾱ_t = f(t)/f(0), f(t) = cos((t/T + s)/(1+s)·π/2)^2.
    - ᾱ maleje od ~1 do ~0: "ile sygnału zostało po t krokach",
    - β_t = 1 - ᾱ_t/ᾱ_{t-1}: "ile sygnału ginie w JEDNYM kroku",
    - s = 0.008 spłasza start — pierwsze kroki prawie nic nie niszczą,
    - clamp β <= beta_max chroni koniec harmonogramu (bez clampu β > 1
      i ᾱ zmieniłaby znak — patrz test_schedule_shapes_and_bounds).
    """
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    abar = f / f[0]                       # abar[0] = 1.0, abar[T] ~ 0
    betas = 1 - abar[1:] / abar[:-1]      # β_t = 1 - ᾱ_t/ᾱ_{t-1}
    betas = np.clip(betas, a_min=0, a_max=beta_max)
    alphas_cumprod = np.cumprod(1 - betas)      # długość T, używane w q_sample
    return dict(betas=betas, alphas_cumprod=alphas_cumprod)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, acp: torch.Tensor):
    """Forward proces JEDNYM ruchem: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε.

    Zamienia czystą linię (x0) w zaszumioną na poziomie t — bez pętli po
    krokach. Po tym dyfuzor trenuje: dla (x_t, t) odgaduje ε.
    ε (noise) zawsze z zewnątrz (torch.Generator z seedem = determinizm);
    t to INDEKS do acp [0..T-1] — z ułamka zamieniaj przez t_frac_to_idx.
    x0/noise: (B, 1, W) albo (B, 1, 32, W) — ᾱ dopasowuje liczbę osi do x0.
    """
    abar_t = torch.as_tensor(acp, dtype=x0.dtype)[t]
    while abar_t.dim() < x0.dim():
        abar_t = abar_t.unsqueeze(-1)     # broadcast po osiach batcha (test: B>1)
    return abar_t.sqrt() * x0 + (1 - abar_t).sqrt() * noise


def t_frac_to_idx(t_frac: float, T: int = 300) -> int:
    """Ułamek harmonogramu -> indeks tablic ᾱ: 0.0 -> 0 (czysta), 1.0 -> T-1 (szum).

    t_frac to język ludzi i configów (np. t_degr_grid w diffusion.yaml);
    t_idx — język tablic. Jedyna dozwolona zamiana między nimi
    (docs/contracts.md, sekcja 2).
    """
    return int(round(t_frac * (T - 1)))
