"""Testy jednostkowe LineDataset (D-tor): wzór resize'u z crnn-ocr, kształt/dtype,
zakres [0,1], sortowanie, determinizm. Wszystko na tmp_path — bez data/raw
(styl jak crnn-ocr/tests/test_dataset.py).

Uruchomienie: python -m pytest tests/test_line_dataset.py

Kontrakt: docs/contracts.md, sekcja 4 (Loader linii).
"""
import torch
from PIL import Image

from data.line_dataset import TARGET_H, LineDataset


def _write_line(d, name, w_px, h_px, fill=255):
    """Pusta linia PIL — kształt i skala wystarczają do testów kontraktu."""
    Image.new("L", (w_px, h_px), fill).save(d / f"{name}.png")


def test_empty_dir(tmp_path):
    assert len(LineDataset(str(tmp_path))) == 0


def test_only_pngs_listed_sorted_txt_ignored(tmp_path):
    _write_line(tmp_path, "b", 64, 64)
    _write_line(tmp_path, "a", 64, 64)
    (tmp_path / "c.txt").write_text("tekst ignorowany", encoding="utf-8")
    ds = LineDataset(str(tmp_path))
    assert [f.name for f in ds.files] == ["a.png", "b.png"]   # sorted, png-only


def test_getitem_contract_shape_dtype_range(tmp_path):
    _write_line(tmp_path, "p", 64, 64, fill=128)              # 64x64 -> resize 32x32
    item = LineDataset(str(tmp_path))[0]
    img = item["image"]
    assert img.shape == (1, TARGET_H, 32)
    assert img.dtype == torch.float32
    assert 0.0 <= float(img.min()) <= float(img.max()) <= 1.0
    assert item["width"] == 32
    assert item["path"].endswith("p.png")


def test_resize_formula_matches_crnn_dataset(tmp_path):
    # wzór crnn-ocr: W = max(1, round(w * 32 / h)); 128x16 -> round(256) = 256
    _write_line(tmp_path, "wide", 128, 16)
    item = LineDataset(str(tmp_path))[0]
    assert item["image"].shape == (1, TARGET_H, 256)
    assert item["width"] == 256


def test_resize_min_width_clamp(tmp_path):
    # 1x100: round(1 * 32/100) = 0 -> clamp do 1 (W nigdy nie jest zerem)
    _write_line(tmp_path, "thin", 1, 100)
    item = LineDataset(str(tmp_path))[0]
    assert item["image"].shape == (1, TARGET_H, 1)
    assert item["width"] == 1


def test_determinism_between_loads(tmp_path):
    # ten sam plik czytany dwukrotnie (także przez osobne instancje) — bajt w bajt
    _write_line(tmp_path, "p", 100, 40, fill=100)
    ds = LineDataset(str(tmp_path))
    assert torch.equal(ds[0]["image"], ds[0]["image"])
    assert torch.equal(ds[0]["image"], LineDataset(str(tmp_path))[0]["image"])
