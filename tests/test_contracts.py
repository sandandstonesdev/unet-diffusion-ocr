"""Testy kontraktów styków (docs/contracts.md, sekcja 8 — tabela identyfikowalności).

Aktywne są kontrakty z implementacją; kontrakty z przyszłych kamieni mają
pełne ciało testu + pytest.mark.skip z powodem — po implementacji usuwamy
marker i test staje się bramką. Żaden test nie zalega czerwony "na zapas".

Uruchomienie: python -m pytest tests/test_contracts.py
(albo python -m tests.test_contracts)
"""
import pytest
import torch
from PIL import Image

from data.line_dataset import TARGET_H, LineDataset


def test_styk_dataset_dyfuzor(tmp_path):
    # kontrakt obrazka (sekcja 1) sprawdzone na tym, co realnie wydaje loader
    Image.new("L", (128, 16), 120).save(tmp_path / "line.png")
    x = LineDataset(str(tmp_path))[0]["image"]
    assert x.shape[0] == 1 and x.shape[1] == TARGET_H          # (1, 32, W)
    assert x.dtype == torch.float32
    assert 0.0 <= float(x.min()) <= float(x.max()) <= 1.0      # zakres [0,1]


def test_degrade_g_contract():
    from configs.schedule import cosine_schedule
    from data.degrade import degrade_g

    x0 = torch.rand(1, 1, 32, 128)                              # [0,1] wejście
    out = degrade_g(x0, t_frac=0.5, schedule=cosine_schedule(300), seed=7)
    assert out.shape == x0.shape                                # W niezmienna
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0   # powrót do [0,1]
    # determinizm: ten sam seed -> bajtowo identyczny wynik
    again = degrade_g(x0, t_frac=0.5, schedule=cosine_schedule(300), seed=7)
    assert torch.equal(out, again)
    # kształt jak z loadera ((1,32,W), bez osi batcha) też przechodzi
    line = torch.rand(1, 32, 96)
    out3 = degrade_g(line, t_frac=0.3, schedule=cosine_schedule(300), seed=1)
    assert out3.shape == line.shape
    assert 0.0 <= float(out3.min()) <= float(out3.max()) <= 1.0


def test_degrade_r_contract():
    from typing import get_args

    from data.degrade import RKind, degrade_r

    x0 = torch.rand(1, 1, 32, 128)
    for kind in get_args(RKind):
        out = degrade_r(x0, kind, seed=7)
        assert out.shape == x0.shape                            # W niezmienna
        assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0
        assert torch.equal(out, degrade_r(x0, kind, seed=7))    # determinizm
    # kontrast dotyka tylko atramentu: czyste tło (1.0) zostaje dokładnie 1.0
    page = torch.ones(1, 1, 32, 128)
    page[..., 8:24] = 0.2                                       # "atrament"
    out = degrade_r(page, "contrast", seed=7)
    assert torch.all(out[page >= 1.0] == 1.0)


@pytest.mark.skip(reason="D2: restore — implementacja w pipeline/restore.py")
def test_restore_passthrough_contract():
    from pipeline.restore import restore

    x = torch.rand(1, 1, 32, 128)                               # zdegradowana [0,1]
    out = restore(x, t0=0.3)
    assert out.shape == x.shape                                  # ta sama W
    assert out.dtype == x.dtype
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0   # powrót do [0,1]


@pytest.mark.skip(reason="S2: dewarp — implementacja w pipeline/dewarp.py")
def test_dewarp_contract():
    import numpy as np

    from pipeline.detect import DetectedLine
    from pipeline.dewarp import dewarp

    # surowy wycinek (uint8, bez normalizacji) — jak z realnej strony
    raw = np.full((20, 100), 255, dtype=np.uint8)
    raw[8:16, :] = 40
    line = DetectedLine(image=raw, bbox=(0, 0, 100, 20), baselines=(0, 100),
                        score=1.0, truncated=False)
    out = dewarp(line)
    assert out.shape[0] == 1 and out.shape[1] == TARGET_H       # (1, 32, W)
    assert out.dtype == torch.float32
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0   # właściciel [0,1]


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v"]))
