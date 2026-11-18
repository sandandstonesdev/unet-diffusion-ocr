"""Loader czystych linii do treningu dyfuzora (D-tor).

Linie czyste (1, 32, W) [0,1] BEZ par — preprocessing identyczny
z LineDataset z crnn-ocr (data/dataset.py), jedno źródło prawdy.

Trening: data/raw/train (ocr-pl-lines train). Ewaluacja: data/raw/val
po degradacjach G/R. Test realny: data/raw/ehri-test (ehri-pl-lines,
ortografia historyczna — przed ΔCER filtr charset 84 klas crnn-ocr).
Ścieżki: configs/diffusion.yaml.

TODO (D0): implementacja loadera + par (degradowana, czysta).
"""
