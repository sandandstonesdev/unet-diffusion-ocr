# unet-diffusion-ocr

## Co to jest

U-Net detekcyjny (podział skanu na linie) + dewarp + dyfuzyjny restaurator linii (SDEdit).
**Zakres: od skanu do naprawionego obrazka linii** 

[crnn-ocr](https://github.com/sandandstonesdev/crnn-ocr) daje cały pipeline do tekstu:

```
skan → detekcja (U-Net linii) → dewarp → linia (1, 32, W) [0,1]
                                             │
                                             ▼
                                 [DYFUZOR: restauracja SDEdit]
                                             │
                                             ▼
                                          CRNN → tekst          (crnn-ocr)
```

**Status: WIP** — gotowe: struktura repo + kontrakty modułów (ten plik).
W trakcie: adapter danych + degradacje (D0). Planowane: trening dyfuzora (D1),
pętla `restore()` (D2), ewaluacja ΔCER (D3), detekcja (S0–S2).

Dwa osobne modele, trenowane niezależnie (wspólna tylko klasa sieci `model/unet.py`):

| | U-Net detekcyjny (S-tor) | U-Net dyfuzyjny (D-tor) |
|---|---|---|
| uczy się | maska bazowej linii per piksel | ε-prediction: „jaki szum dodano do linii" |
| loss | BCE + Dice | MSE na szumie + harmonogram T=300 |
| dane | strony renderowane + maski GT (`S0`) | czyste linie z crnn-ocr, bez par |
| trening | `train_detect.py` → `checkpoints/detect/` | `train_unet.py` → `checkpoints/diffusion/` |
| inference | raz na stronę: skan → `DetectedLine` | opcjonalnie per linia: `restore()` |

Kolejność trenowania dowolna — jeden moduł od zera naraz. Tor D nie czeka na tor S
(`splitter_fallback.py` albo linie prosto z `crnn-ocr/data/raw/`).

## Struktura katalogów

```
unet-diffusion-ocr/
│
├── README.md                  # ten plik
├── environment.yml            # specyfikacja środowiska conda
├── .gitignore
│
├── configs/
│   ├── detect.yaml            # hiperparametry detektora (S-tor)
│   ├── diffusion.yaml         # hiperparametry dyfuzora (D-tor)
│   └── schedule.py            # harmonogram cosinusowy + q_sample (tylko dyfuzor)
│
├── data/
│   ├── __init__.py
│   ├── mask_dataset.py        # strony + maski bazowych linii (S0)
│   ├── line_dataset.py        # czyste linie (1,32,W) [0,1] do treningu dyfuzora
│   ├── degrade.py             # G-degradacja (q_sample, siatka t_degr) + R-degradacje
│   └── splitter_fallback.py   # v0: profilowanie rzędów (testy bez U-Net detekcyjnego)
│
├── model/
│   ├── __init__.py
│   ├── unet.py                # klasa U-Net (M1: 3 piętra, base 32, GN(8)+SiLU) — dla obu modeli
│   └── time_embed.py          # embedding t (warunkowanie tylko dyfuzora)
│
├── pipeline/
│   ├── __init__.py
│   ├── detect.py              # strona → maska → DetectedLine (S2)
│   ├── dewarp.py              # resize H=32, prostowanie, skala [0,1]
│   └── restore.py              # pętla SDEdit (DDIM od t0 do 0) + brama/skip (D2)
│
├── metrics.py                 # ΔCER — pary val (zdegradowana) vs val (restaurowana)
├── train_detect.py            # trening detektora: maski, BCE+Dice (S1)
├── train_unet.py              # trening dyfuzora: ε-prediction, MSE, EMA (D1)
├── evaluate.py                # ΔCER + mapa 2D (t_degr, t0) — D3
├── infer.py                   # skan → lista naprawionych linii (PNG per linia)
│
├── runs/                      # logi TensorBoard (IGNOROWANE)
├── checkpoints/               # modele (IGNOROWANE): detect/, diffusion/
├── data/raw/                  # dane pobierane osobno (IGNOROWANE)
└── tests/                     # testy kontraktów: kształt, [0,1], geometria baseline
```

`data/raw/`, `runs/`, `checkpoints/` ignorowane przez git — dane i artefakty, nie kod.

## Kontrakty

Od dewarpu w prawo jedna konwencja: **[0,1] float32, H = 32** (dziedziczone z
`data/dataset.py` w crnn-ocr — jedno źródło prawdy). Padding zerami z prawej robi
dopiero `collate` CRNN — restaurator go nie dotyka. Kontrakty:

- `DetectedLine` — surowy wycinek skanu (splitter nie skaluje i nie normalizuje),
- `dewarp` → `(1, 32, W) float32 [0,1]` — właściciel konwencji,
- `restore` — przelotowy: ta sama W, ta sama geometria baseline'u; wewnętrznie
  pracuje w [-1,1], konwersje zamknięte w środku.

```python
@dataclass
class DetectedLine:
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    baselines: tuple[int, int]
    score: float
    truncated: bool

def dewarp(line: DetectedLine) -> Tensor:
    ...

def restore(degraded: Tensor, t0: float | None = None) -> Tensor:
    ...
```

## Kamienie milowe

| Tor | Zadanie | Definicja gotowości |
|---|---|---|
| D0 | adapter danych + degradacje G/R | loader linii (= LineDataset z crnn-ocr); 200 próbek siatki G |
| D1 | trening U-Net dyfuzyjnego | próbki DDIM = spójne linie; overfit-check na 1 batchu |
| D2 | pętla SDEdit `restore()` | działa na 1 zdegradowanej linii; siatka t0 {0.2, 0.3, 0.5, 0.7} |
| D3 | ewaluacja ΔCER | raport na CRNN z crnn-ocr; decyzja: włącz / odłóż / iteruj (v2) |
| S0 | renderer: linie → strony + maski GT | maski spójne z renderem |
| S1 | trening U-Net detekcyjnego | maska spójna na stronach testowych; overfit-check |
| S2 | postprodukcja: maska → `DetectedLine` | bboxy vs 10–20 ręcznie anotowanych linii |

## Dane

Dwa datasety (styl crnn-ocr: trening / ewaluacja / test):

| | Dyfuzor (D) | Detektor (S) |
|---|---|---|
| trening | [ocr-pl-lines](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines) train — te same linie co CRNN | strony renderowane w S0 z linii train (maski GT za darmo) |
| ewaluacja | ocr-pl-lines val po degradacjach G/R | strony renderowane z linii val |
| test | [ehri-pl-lines](https://huggingface.co/datasets/PiotrSty/ehri-pl-lines) test — 81 realnych linii | ehri-pl-lines — porównanie bboxów z kilkoma anotacjami |

Pary `{base}.png` + `{base}.txt` pobierz osobno (`snapshot_download`) i umieść w:

```
data/raw/train/        # ocr-pl-lines train (te same pliki co crnn-ocr/data/raw/train)
data/raw/val/          # ocr-pl-lines val
data/raw/ehri-test/    # ehri-pl-lines test
```

Uwaga: ehri-pl-lines to ortografia historyczna — przed ΔCER przefiltruj teksty przez
charset 84 klas z crnn-ocr. Atrybucja: ehri-pl-lines — EHRI / Archiwum ŻIH, CC-BY 4.0;
ocr-pl-lines — MIT.

## Setup

```bash
conda env create -f environment.yml
conda activate unet_diffusion_ocr
python -m tests.test_contracts    # sanity: kształty, [0,1], geometria baseline (od D0)
```

## Integracja z crnn-ocr

`infer.py` zwraca naprawione linie `(1, 32, W)` [0,1] — kontrakt identyczny z wyjściem
dewarpa, więc CRNN (collate → CRNN → greedy CTC) konsumuje bez zmian w loaderze.
Najprostszy retrofit: restore na `data/raw/val` → zapis do `data/restored/val/` →
ta sama ewaluacja CRNN na obu katalogach.

## Wyniki

*(do uzupełnienia po pierwszych treningach: ΔCER(t_degr, t0) na degradacji G,
mapa 2D, kontrola halucynacji na liniach czystych, test realny na ehri-pl-lines)*

## Licencja

Kod projektu: licencja **MIT** — pełny tekst w [LICENSE](https://github.com/slayerlabs/crnn-ocr/blob/main/LICENSE).

Dane (`data/raw/`) nie są częścią repozytorium. Zbiór [ocr-pl-lines](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines)
jest udostępniony na licencji **MIT** (zgodnie z kartą datasetu), więc publikacja
artefaktów treningu nie jest ograniczona. Zbiór [ehri-pl-lines](https://huggingface.co/datasets/PiotrSty/ehri-pl-lines)
— licencja **CC-BY 4.0** (EHRI / Archiwum ŻIH): wymagana atrybucja; materiał
historycznie wrażliwy, używany wyłącznie do badań OCR/HTR, nie jest redystrybuowany.

## Autorzy

* **Paweł Aksiutowicz** — Lead Developer

## Podziękowania

* Generowanie i optymalizacja kodu przy wsparciu **Mistral AI**

* Szczególne podziękowania za nieocenione zasoby, kod i mocne wsparcie w nauce:

  * Kacper Wikieł [@kwikiel](https://github.com/kwikiel/)

  * Arkadiusz Słota [@Maggio333(https://github.com/Maggio333)]

  * **Piotr Styla** [@PiotrStyla](https://github.com/PiotrStyla/) — [zbiór linii treningowych](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines) oraz [ehri-pl-lines](https://huggingface.co/datasets/PiotrSty/ehri-pl-lines)

  * **Slayer AI Labs** [@slayerlabs](https://github.com/slayerlabs)
