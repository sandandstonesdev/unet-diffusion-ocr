"""Dewarp: właściciel konwencji [0,1] float32 i H=32.

Wzór resize'u z LineDataset (crnn-ocr), prostowanie wg baselines."""

from torch import Tensor

from .detect import DetectedLine


def dewarp(line: DetectedLine) -> Tensor:
    """(1, 32, W) float32 [0,1] — kontrakt wyjściowy na całym D-tracku."""
    raise NotImplementedError("S-track / D0: dewarp")
