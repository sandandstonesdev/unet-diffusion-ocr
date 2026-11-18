"""Dane detektora (S0): strony + maski bazowych linii z renderu.

Trening: strony renderowane z linii train ocr-pl-lines (maski GT
generują się automatycznie przy renderze). Ewaluacja: strony renderowane
z linii val (kompozycja niedostępna w treningu). Ścieżki: configs/detect.yaml.

TODO (S0):
    - renderer masek GT: maski spójne z renderem; zapis par (strona, maska)
"""
