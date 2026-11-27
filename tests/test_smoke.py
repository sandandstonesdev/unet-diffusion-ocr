"""Smoke test: cały flow na syntetycznej linii, krok po kroku, z in/out.

Uruchomienie: python -m tests.test_smoke    (czytelny wydruk in/out każdego etapu)
          albo: python -m pytest tests/     (ten sam kod jako cicha bramka)

Zero zależności od data/raw — linia syntetyczna PIL-em (jak w notebookach).
Mock = MECHANIKA, nie jakość: model dyfuzora to 2 conv-y, trening to 1 krok
ε-prediction. Ten plik rośnie razem z pipeline: D0 wpięło degrade_g (sekcja 4),
po D1 podmieniamy MockEps na model/unet.py, po D2 dochodzi restore(), po S2 dewarp().
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from configs.schedule import cosine_schedule, q_sample, t_frac_to_idx
from data.degrade import degrade_g

T = 300


def _p(verbose, *args):
    if verbose:
        print(*args)


def _synth_line(text, w=256, h=32):
    """Linia syntetyczna PIL-em -> (1, 1, 32, W) float32 [0,1] (jak notebooki)."""
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    im = Image.new("L", (w, h), 255)
    ImageDraw.Draw(im).text((6, 2), text, fill=30, font=font)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


class MockEps(nn.Module):
    """Mock dyfuzora: 2 conv-y, t doklejone jako kanał. To NIE model/unet.py.

    Cel: pokazać mechanikę ε-prediction (dostaje (x_t, t), zwraca ε̂) na 1 kroku.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1), nn.SiLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, xt, t_frac):        # t_frac: (B,) float [0,1] — ułamek!
        b = t_frac.view(-1, 1, 1, 1)
        b = b.expand(-1, 1, xt.shape[2], xt.shape[3])
        return self.net(torch.cat([xt, b], dim=1))


def _flow(verbose):
    stats = {}
    _p(verbose, "=" * 64)
    _p(verbose, "SMOKE FLOW: harmonogram -> linia -> G -> 1 krok mock-treningu")
    _p(verbose, "=" * 64)

    # -- 1. harmonogram ------------------------------------------------------
    sched = cosine_schedule(T)
    acp = torch.as_tensor(sched["alphas_cumprod"], dtype=torch.float32)
    _p(verbose, "")
    _p(verbose, "[1] harmonogram: cosine_schedule(300)")
    _p(verbose, f"    ᾱ[0.0]={float(acp[0]):.4f}  "
                f"ᾱ[0.5]={float(acp[t_frac_to_idx(0.5)]):.4f}  "
                f"ᾱ[1.0]={float(acp[-1]):.5f}")
    stats["abar_first"] = float(acp[0])
    stats["abar_last"] = float(acp[-1])

    # -- 2. linia wejściowa (format loadera: [0,1]) --------------------------
    batch = torch.cat([_synth_line("Zażółć gęślą jaźń"),
                       _synth_line("Strzeżona zagroda")])    # (2, 1, 32, W)
    _p(verbose, "")
    _p(verbose, "[2] linie wejściowe (jak z line_dataset: [0,1])")
    _p(verbose, f"    in:  kształt={tuple(batch.shape)}  "
                f"zakres=[{float(batch.min()):.2f}, {float(batch.max()):.2f}]")
    stats["x_shape"] = tuple(batch.shape)
    stats["x_min"], stats["x_max"] = float(batch.min()), float(batch.max())

    # -- 3. wnętrze dyfuzora ([-1,1]) ----------------------------------------
    x0 = 2 * batch - 1
    _p(verbose, "")
    _p(verbose, "[3] konwersja do wnętrza dyfuzora: x0 = 2x - 1")
    _p(verbose, f"    in:  zakres=[{float(x0.min()):.2f}, {float(x0.max()):.2f}]")

    # -- 4. degradacja G (degrade_g z data/degrade.py; ε z zewnątrz) --------
    g = torch.Generator().manual_seed(42)
    t_idx = torch.tensor([t_frac_to_idx(0.5), t_frac_to_idx(0.35)])
    eps = torch.randn(x0.shape, generator=g)          # cel ε-prediction w [5]
    xt = q_sample(x0, t_idx, eps, acp)                # para treningowa (x_t, ε)
    deg = degrade_g(batch, t_frac=torch.tensor([0.5, 0.35]),
                    schedule=sched, noise=eps)
    assert torch.allclose(deg, ((xt + 1) / 2).clamp(0, 1))  # moduł == ręczna matematyka
    _p(verbose, "")
    _p(verbose, "[4] degradacja G: degrade_g(batch, t=[0.5, 0.35], noise=ε)")
    _p(verbose, f"    out: kształt={tuple(deg.shape)}  "
                f"zakres=[{float(deg.min()):.2f}, {float(deg.max()):.2f}]"
                "  (clamp na brzegu)")
    stats["deg_shape"] = tuple(deg.shape)
    stats["deg_min"], stats["deg_max"] = float(deg.min()), float(deg.max())

    # -- 5. 1 krok mock-treningu (ε-prediction) -----------------------------
    torch.manual_seed(42)         # determinizm całego kroku (wagi mocka + eps)
    model = MockEps()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred = model(xt, t_idx.float() / (T - 1))              # t jako UŁAMEK do modelu
    loss = F.mse_loss(pred, eps)                          # ε̂ vs ε (nie vs x_t!)
    opt.zero_grad()
    loss.backward()
    grad = sum(p.grad.abs().sum().item()
               for p in model.parameters() if p.grad is not None)
    loss_val = float(loss.detach())
    opt.step()
    _p(verbose, "")
    _p(verbose, "[5] 1 krok mock-treningu: MSE(ε̂, ε) + backward + step")
    _p(verbose, f"    out: loss={loss_val:.4f}  |∇|={grad:.1f}  "
                "(loss finite, gradienty niezerowe = mechanika działa)")
    stats["loss"] = loss_val
    stats["grad"] = grad
    _p(verbose, "")
    return stats


def test_flow_smoke():
    stats = _flow(verbose=False)
    # kontrakty etapów (docs/contracts.md, sekcje 1-3)
    assert stats["x_shape"] == (2, 1, 32, 256)
    assert 0.0 <= stats["x_min"] <= stats["x_max"] <= 1.0
    assert stats["abar_first"] > 0.99 and stats["abar_last"] < 0.01
    assert stats["deg_shape"] == stats["x_shape"]           # W niezmienna
    assert 0.0 <= stats["deg_min"] <= stats["deg_max"] <= 1.0
    assert np.isfinite(stats["loss"]) and stats["loss"] > 0
    assert stats["grad"] > 0


if __name__ == "__main__":
    _flow(verbose=True)
    print("OK — smoke flow przeszedł")
