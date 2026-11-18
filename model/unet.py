"""JEDNA klasa U-Net (M1) dla obu torów — detektora (S) i dyfuzora (D).

Szkielet 3 pięter, base 32, skip connections, GroupNorm(8)+SiLU.
Warunkowanie czasem t NIE jest częścią backbone'a — embedding dostarcza
model/time_embed.py i używa go tylko dyfuzor.

TODO:
    - S1: detektor — wejście strona, wyjście maska bazowej linii per piksel
    - D1: dyfuzor — ε-prediction na (1, 32, W) [-1,1] wewnętrznie
"""
