"""Testy harmonogramu dyfuzyjnego — port asercji z notebooks/d0_schedule_inspect
plus regresje z pierwszej sesji (broadcast batcha B>1).

Uruchomienie: python -m pytest tests/test_schedule.py

Kontrakt: docs/contracts.md, sekcje 2-3 (Czas, Harmonogram).
"""
import numpy as np
import torch

from configs.schedule import cosine_schedule, q_sample, t_frac_to_idx

T = 300


def _acp() -> torch.Tensor:
    return torch.as_tensor(cosine_schedule(T)["alphas_cumprod"], dtype=torch.float32)


def test_schedule_shapes_and_bounds():
    sched = cosine_schedule(T)
    acp, betas = sched["alphas_cumprod"], sched["betas"]
    assert len(acp) == T and len(betas) == T
    assert acp[0] > 0.99               # ᾱ[0] = 1 - β1 — nigdy dokładnie 1
    assert acp[-1] < 0.01              # na końcu (prawie) czysty szum
    assert np.all(np.diff(acp) < 0)     # ᾱ ściśle malejąca
    assert betas.max() <= 0.999         # clamp β


def test_t_frac_to_idx_extremes():
    assert t_frac_to_idx(0.0, T) == 0
    assert t_frac_to_idx(1.0, T) == T - 1
    idx = [t_frac_to_idx(f, T) for f in np.linspace(0.0, 1.0, 11)]
    assert idx == sorted(idx)           # monotoniczne mapowanie


def test_q_sample_at_t_zero_keeps_x0():
    # ᾱ[0] = 1 - β1, więc q_sample(t=0) trzyma x0 z tolerancją ~2e-3, nie 1e-5
    x0 = torch.rand(1, 1, 32, 64, generator=torch.Generator().manual_seed(1)) * 2 - 1
    out = q_sample(x0, torch.tensor([0]), torch.zeros_like(x0), _acp())
    assert torch.allclose(out, x0, atol=2e-3)


def test_q_sample_at_t_max_kills_signal():
    g = torch.Generator().manual_seed(42)
    x0 = torch.rand(1, 1, 32, 256, generator=g) * 2 - 1
    noise = torch.randn(1, 1, 32, 256, generator=g)
    xt = q_sample(x0, torch.tensor([T - 1]), noise, _acp())
    corr = torch.corrcoef(torch.stack([xt.flatten(), x0.flatten()]))[0, 1]
    assert abs(float(corr)) < 0.2       # czysty szum: brak korelacji z x0


def test_q_sample_matches_formula_manual():
    # dokładna matematyka: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε policzona wprost
    acp = _acp()
    t = torch.tensor([0, 75, 150, 299])
    x0 = torch.rand(4, 1, 32, 16, generator=torch.Generator().manual_seed(7)) * 2 - 1
    eps = torch.randn(4, 1, 32, 16, generator=torch.Generator().manual_seed(8))
    xt = q_sample(x0, t, eps, acp)
    a = acp[t].view(-1, 1, 1, 1)
    manual = a.sqrt() * x0 + (1 - a).sqrt() * eps
    assert torch.allclose(xt, manual, atol=1e-6)


def test_q_sample_batch_broadcast_regression():
    # regresja: ᾱ musi rozciągać się po osiach batcha; (B,1,32,W) bez rozdęcia
    x0 = torch.rand(4, 1, 32, 64) * 2 - 1
    xt = q_sample(x0, torch.tensor([10, 20, 30, 40]), torch.randn_like(x0), _acp())
    assert xt.shape == (4, 1, 32, 64)


def test_q_sample_determinism_with_seed():
    # noise zawsze z zewnątrz → determinizm = ten sam generator (docs/contracts.md)
    x0 = torch.rand(1, 1, 32, 64) * 2 - 1
    t = torch.tensor([150])
    n1 = torch.randn(x0.shape, generator=torch.Generator().manual_seed(7))
    n2 = torch.randn(x0.shape, generator=torch.Generator().manual_seed(7))
    assert torch.equal(q_sample(x0, t, n1, _acp()), q_sample(x0, t, n2, _acp()))
