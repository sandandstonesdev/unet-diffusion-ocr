# D0 — siatka degradacji G (wpis bramki)

**Status: bramka D0 spełniona** (2026-11-27). Definicja gotowości (README,
kamienie milowe): *200 próbek siatki t_degr*.

## Przebieg

- **Źródło linii**: `data/raw/val` — ocr-pl-lines val, 200 par PNG+TXT
  (`snapshot_download` z `PiotrSty/ocr-pl-lines`); wybór 20 linii
  deterministyczny: `sorted(random.Random(seed=42).sample(range(200), 20))`.
- **Poziomy**: `t_frac` = 0.0 (odniesienie) + `t_degr_grid` 0.1–0.9
  (`configs/diffusion.yaml`) → 10 kolumn.
- **Siatka**: 20 linii × 10 poziomów = **200 próbek**; batch = 1 per próbka
  (W zmienna per linia — bez paddingu, kontrakt sekcji 1); seed per próbka
  (`seed + r*10 + c`), więc manifest odtwarza każdą próbkę niezależnie.
- **Skrypt**: `python make_d0_grid.py` (opcje `--synthetic`, `--limit N`
  do testu mechaniki bez danych).

## Wynik

| Artefakt (`runs/d0_grid/`, IGNOROWANE przez git) | sha256 |
|---|---|
| `manifest.jsonl` (200 wpisów: line, out, t_frac, seed, sha256) | `3ae4cdd7…8231` |
| `contact_sheet.png` (arkusz 20×10 — notatka, nie bramka) | `2ba7b4d2…f05b` |
| `samples/` (200 PNG, uint8) | 200/200 unikalnych sha256 |

Kontrola odtwarzalności wbudowana w skrypt: ponowne wygenerowanie pierwszej
próbki z `(line, t_frac, seed)` z manifestu daje identyczny PNG (sha zgodne).

## Decyzje na D1

- Udział sygnału w próbce (√ᾱ, z harmonogramu): t=0.1 → 0.985, t=0.3 → 0.885,
  t=0.5 → 0.699, t=0.7 → 0.451, t=0.9 → 0.155. Czytelność linii spada
  wyraźnie od ~t=0.5 (szum ≥ sygnał), czysty szum przy t=0.9 — siatka
  potwierdza to na 20 liniach val (contact-sheet).
- `degrade_g` przyjmuje ε z zewnątrz (`noise=`) — trening D1 paruje
  (x_t, ε) bez powtórnego losowania, przez `q_sample` jak w smoke-flow.
