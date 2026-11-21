"""Loader czystych linii do treningu dyfuzora (D-tor).

Linie czyste (1, 32, W) [0,1] BEZ par — preprocessing identyczny
z LineDataset z crnn-ocr (data/dataset.py), jedno źródło prawdy: ten sam
wzór resize'u, ta sama skala, deterministyczne sortowanie plików.
Pliki .txt są ignorowane (D-tor trenuje bez tekstu); filtry CRNN (charset,
CTC) nic tu nie znaczą — parowanie z CRNN jest po stronie ewaluacji ΔCER (D3).

Trening: data/raw/train (ocr-pl-lines train). Ewaluacja: data/raw/val
po degradacjach G/R. Test realny: data/raw/ehri-test (ehri-pl-lines,
ortografia historyczna — przed ΔCER filtr charset 84 klas crnn-ocr).
Ścieżki: configs/diffusion.yaml.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGET_H = 32          # jak data/dataset.py w crnn-ocr — jedno źródło prawdy


class LineDataset(Dataset):
    """Linia PNG → {"image", "path", "width"}; image: (1, 32, W) float32 [0,1].

    Transformacja identyczna z crnn-ocr/data/dataset.py — jedno źródło prawdy
    (kontrakt: docs/contracts.md, sekcja 4; testy: tests/test_line_dataset.py).
    """

    def __init__(self, root: str, w_bucket: int = 0):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.png"))   # deterministyczna kolejność
        self.w_bucket = w_bucket      # TODO (D1): bucketing szerokości w treningu

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, path) -> torch.Tensor:
        """PNG -> (1, 32, W) float32 [0,1]; W = max(1, round(w*32/h)) jak crnn-ocr."""
        img = Image.open(path).convert("L")
        w = max(1, round(img.width * TARGET_H / img.height))
        arr = np.asarray(img.resize((w, TARGET_H)), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, i) -> dict:   # zwraca dict, nie goły tensor
        img = self._load(self.files[i])                  # (1,32,W) [0,1] float32
        return {"image": img, "path": str(self.files[i]), "width": int(img.shape[-1])}
