"""Degradacje: G (dyfuzyjna, q_sample) oraz R (realistyczne zakłócenia).

G to degradacja "modelowa": dokładnie ten sam forward process, przeciw
któremu trenuje dyfuzor — na G stoi siatka t_degr (D0, make_d0_grid.py)
i ewaluacja ΔCER (D3). R to "świat realny": klasyczne transformacje
obrazkowe (blur, szum, prześwity, JPEG) jako alternatywa/uzupełnienie G.

Wspólne kontrakty (docs/contracts.md, sekcje 1 i 5; testy: test_contracts.py):
- wejście/wyjście [0,1] float32, W bez zmian (degradacje są przelotowe),
- kształt zachowany: (1,32,W) albo (B,1,32,W),
- determinizm: ten sam seed -> bajtowo identyczny wynik (torch.Generator).
"""

from io import BytesIO
from typing import Literal, get_args

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from configs.schedule import q_sample, t_frac_to_idx

# Rodzaje degradacji R — katalog wartości; degrade_r odrzuca wszystko poza nim.
RKind = Literal[
    "blur_gauss", "blur_motion", "noise_gauss", "salt_pepper",
    "bleed_through", "jpeg", "contrast",
]


def degrade_g(x0_clean: Tensor, t_frac, schedule, seed=None, noise=None) -> Tensor:
    """G: czysta linia -> zaszumiona jednym ruchem (forward process).

    [0,1] -> [-1,1] (x2 = 2x-1) -> q_sample -> (xt+1)/2 z clampem -> [0,1].
    Konwersje [0,1] <-> [-1,1] tylko tutaj, na brzegu (kontrakt sekcji 1).

    t_frac: float dla całego tensora albo (B,) per linia batcha; jedyna
    dozwolona zamiana na indeks ᾱ to t_frac_to_idx (kontrakt sekcji 2).
    noise: ε z zewnątrz, jak w q_sample — np. parowanie (x_t, ε) w smoke/D1;
    None -> torch.randn z torch.Generator(seed). Ten sam seed -> bajtowo
    identyczny wynik (kontrakt sekcji 5). Kształt noise: (B,1,32,W).
    """
    acp = torch.as_tensor(schedule["alphas_cumprod"], dtype=torch.float32)
    T = acp.shape[0]
    if isinstance(t_frac, Tensor):
        t_idx = torch.tensor([t_frac_to_idx(float(t), T) for t in t_frac])
    else:
        t_idx = torch.tensor([t_frac_to_idx(float(t_frac), T)])

    squeeze = x0_clean.dim() == 3                    # (1,32,W) z loadera
    x0 = x0_clean.unsqueeze(0) if squeeze else x0_clean
    if noise is None:
        g = None if seed is None else torch.Generator().manual_seed(int(seed))
        noise = torch.randn(x0.shape, generator=g)
    xt = q_sample(2 * x0 - 1, t_idx, noise, acp)     # wnętrze dyfuzora: [-1,1]
    out = ((xt + 1) / 2).clamp(0, 1)
    return out.squeeze(0) if squeeze else out


def _jpeg_roundtrip(x: Tensor, quality: int) -> Image.Image:
    """(1,32,W) [0,1] -> JPEG w pamięci (bez pliku) -> obraz PIL z artefaktami."""
    arr = (x[0] * 255).round().clamp(0, 255).byte().numpy()
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("L")


def degrade_r(x0_clean: Tensor, kind: RKind, seed=None) -> Tensor:
    """R: realistyczne brudy jako klasyczne transformacje obrazkowe.

    Operacje na [0,1]; W niezmienna; tło ≠ atramentu nie rozlewa się
    (contrast dotyka tylko atramentu). Siła degradacji losowana z
    torch.Generator(seed) — ten sam seed -> identyczny wynik.

    blur_gauss    — gaussian blur: atrament się rozmywa („plama" tuszu),
    blur_motion   — pozioma smuga (ruch skanu wzdłuż linii),
    noise_gauss   — addytywny szum gaussowski,
    salt_pepper   — pieprz (kropki na tle) + sól (dziury w atramencie),
    bleed_through — prześwit z drugiej strony kartki: odbicie lustrzane
                    liniki z alfa (ciemny atrament prześwituje jako szarość),
    jpeg          — artefakty kompresji (round-trip PIL, losowa jakość),
    contrast      — wyblaknięcie (gamma > 1) / pogrubienie (gamma < 1)
                    atramentu; tło (1.0) zostaje dokładnie 1.0.
    """
    if kind not in get_args(RKind):
        raise ValueError(
            f"nieznana degradacja R: {kind!r}; dozwolone: {get_args(RKind)}")
    squeeze = x0_clean.dim() == 3                    # (1,32,W) z loadera
    x = x0_clean.unsqueeze(0) if squeeze else x0_clean
    g = None if seed is None else torch.Generator().manual_seed(int(seed))

    def _u(lo: float, hi: float) -> float:           # losowa siła degradacji
        return float(lo + (hi - lo) * torch.rand((), generator=g))

    if kind == "blur_gauss":
        sigma = _u(0.8, 1.5)
        k = 2 * int(np.ceil(3 * sigma)) + 1          # nieparzysty
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        k1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        k1 = k1 / k1.sum()
        k2d = k1[:, None] * k1[None, :]             # kernel 2D z dwóch 1D
        out = F.conv2d(x, k2d.view(1, 1, k, k), padding=k // 2)
    elif kind == "blur_motion":
        k = int(_u(3, 7))                            # 3..6 px smugi
        out = F.conv2d(x, torch.ones(1, 1, 1, k) / k, padding=(0, k // 2))
    elif kind == "noise_gauss":
        out = (x + torch.randn(x.shape, generator=g) * _u(0.05, 0.15)).clamp(0, 1)
    elif kind == "salt_pepper":
        p = _u(0.01, 0.03)
        r = torch.rand(x.shape, generator=g)
        out = x.clone()
        out[r < p / 2] = 0.0                         # pieprz: kropki na tle
        out[(r >= p / 2) & (r < p)] = 1.0            # sól: dziury w atramencie
    elif kind == "bleed_through":
        alpha = _u(0.15, 0.35)
        show_through = (1 - x.flip(-1)).clamp(0, 1)  # atrament „z drugiej strony"
        out = (x - alpha * show_through).clamp(0, 1)
    elif kind == "jpeg":
        quality = int(_u(30, 70))
        out = torch.cat([
            torch.from_numpy(
                np.asarray(_jpeg_roundtrip(im, quality), dtype=np.float32) / 255.0
            ).unsqueeze(0).unsqueeze(0)
            for im in x
        ])
    else:                                            # contrast — tylko atrament
        gamma = _u(0.6, 1.8)
        out = 1 - (1 - x).pow(gamma)                  # tło: 1 - 0**gamma = 1.0
    return out.squeeze(0) if squeeze else out
