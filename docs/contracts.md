# Kontrakty modułów

Jedno źródło prawdy konwencji obrazka: `crnn-ocr/data/dataset.py` (loader CRNN —
konsument naszych wyjść). Ten plik jest umową **styku między modułami**;
każdy kontrakt ma test (sekcja 8). Zasada: kontrakt najpierw, implementacja
potem; testy z przyszłych kamieni mają `pytest.mark.skip` z powodem.

## 1. Obrazek linii (jedna konwencja od dewarpa w prawo)

- tensor linii: **`(1, 32, W)` float32, zakres [0,1]**, atrament ≈ 0 (ciemny), tło ≈ 1,
- `H = 32` zawsze; **W zmienna per linia** i niezmienna na całym torze D
  (degradacje i `restore` są przelotowe w szerokość),
- padding zerami z prawej robi **dopiero collate CRNN** — restaurator go nie
  dotyka i nie dodaje paddingu,
- wnętrze dyfuzora pracuje w **[-1,1]** (`x2 = 2·x − 1`); konwersja [0,1] ↔ [-1,1]
  wyłącznie na brzegach funkcji, nigdy w środku rurociągu.

## 2. Czas (tylko dyfuzor)

- `t_frac` ∈ [0,1] — ułamek harmonogramu (ludzie, configi, raporty),
- `t_idx` ∈ {0..T-1} — indeks do tablic ᾱ (kod),
- zamiana **wyłącznie** przez `t_frac_to_idx(t_frac, T)`,
- `T = 300, s = 0.008, beta_max = 0.999` — `configs/diffusion.yaml`.

Klasy: `t_frac` do modelu idzie przez embedding (ułamek), do tablic ᾱ jako
indeks. Pomylenie tych dwóch to klasyczny bug dyfuzora — testowane w
`test_schedule.py`.

## 3. Harmonogram (`configs/schedule.py`) — zaimplementowany

- `cosine_schedule(T=300, s=0.008, beta_max=0.999)` →
  `{"betas": np.ndarray[T], "alphas_cumprod": np.ndarray[T]}`,
- gwarancje: ᾱ ściśle malejąca; `ᾱ[0] = 1 − β₁ > 0.99` (nigdy dokładnie 1 —
  dlatego `q_sample(t=0)` trzyma x0 do tolerancji **atol=2e-3**, nie 1e-5);
  `ᾱ[T-1] < 0.01`; `β ∈ [0, beta_max]`,
- `q_sample(x0, t, noise, acp)`: `x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε`;
  x0/noise: `(B, 1, W)` albo `(B, 1, H, W)` — ᾱ rozciąga się po liczbie osi x0
  (poprawny broadcast batcha; regresja w testach),
- ε (`noise`) zawsze dostarczane z zewnątrz — determinizm przez
  `torch.Generator(seed)`; funkcja sama nie losuje.

## 4. Loader linii (`data/line_dataset.py`) — zaimplementowany

- `LineDataset(root)` indeksuje `*.png` **posortowane** (deterministyczna
  kolejność); pliki `.txt` są ignorowane (D-tor trenuje bez tekstu) —
  składnia bez par jest zamierzona,
- `__getitem__` → `{"image": (1,32,W) float32 [0,1],
  "path": str, "width": int}` (dict, nie goły tensor),
- resize **identyczny** z crnn-ocr: `W = max(1, round(w · 32 / h))`,
  `img.resize((W, 32))` bez jawnego filtra — jak w `crnn-ocr/data/dataset.py`,
- filtry CRNN (charset, CTC) **nie działają** w loaderze dyfuzora; jeśli пара
  ΔCER ma pasować do ewaluacji CRNN, filtrowanie jest po stronie konsumenta (D3),
- TODO (D1): `w_bucket` — bucketing po szerokości dla batchy treningowych
  (padding dopiero w collate CRNN, nie tutaj).

## 5. Degradacje (`data/degrade.py`) — kontrakt, implementacja: D0

- `degrade_g(x0[0,1], t_frac, schedule, seed) -> (1,32,W) [0,1]`:
  wewnątrz [-1,1] → `q_sample` → powrót z clamp; W bez zmian; ten sam seed →
  bajtowo identyczny wynik,
- `degrade_r(x0, kind, seed)`: `kind ∈ RKind` (Literal: blur_gauss, blur_motion,
  noise_gauss, salt_pepper, bleed_through, jpeg, contrast); operacje na [0,1],
  W bez zmian, **tło ≠ atramentu nie rozlewa się** (kontrast dotyka tylko
  atramentu).

## 6. Mostek S → D (`pipeline/`) — kontrakt, implementacja: S0–S2 / D2

- `DetectedLine`: **surowy wycinek strony** (uint8; splitter nie skaluje i nie
  normalizuje), bbox w przestrzeni strony, `baselines=(x_start, x_end)`,
  score 0–1, `truncated` = linia dotyka brzegu,
- `dewarp(DetectedLine) → (1,32,W) float32 [0,1]` — **właściciel konwencji**
  sekcji 1; wzór resize jak loader,
- `restore(x[0,1], t0) → (1,32,W) [0,1]` — przelotowy: ta sama W, ta sama
  geometria baseline (przesunięcie środka ciężaru < 1 px),
- `splitter_fallback` zwraca `DetectedLine` jak detekcja — dewarp i restore
  testowalne bez wytrenowanego toru S.

## 7. Środowisko i uruchomienia

- środowisko: `environment.yml` (torch, numpy, pillow, pyyaml, pytest, ruff,
  matplotlib),
- testy: `python -m pytest tests/` (projektowe) — sanity:
  `python -m tests.test_contracts`,
- lint: `ruff check .` (konfiguracja `ruff.toml` jak w crnn-ocr).

## 8. Tabela identyfikowalności (kontrakt → test → status)

| Kontrakt | Test | Status |
|---|---|---|
| harmonogram: kształty, monotonię, ograniczenia β/ᾱ | `test_schedule.py::test_schedule_shapes_and_bounds` | gotowy |
| czas: `t_frac_to_idx` ekstrema i monotonia | `test_schedule.py::test_t_frac_to_idx_extremes` | gotowy |
| `q_sample`: t=0 trzyma x0 (atol 2e-3) | `test_schedule.py::test_q_sample_at_t_zero_keeps_x0` | gotowy |
| `q_sample`: t=T-1 zabija sygnał | `test_schedule.py::test_q_sample_at_t_max_kills_signal` | gotowy |
| `q_sample`: zgodność ze wzorem | `test_schedule.py::test_q_sample_matches_formula_manual` | gotowy |
| `q_sample`: broadcast batcha (regresja B>1) | `test_schedule.py::test_q_sample_batch_broadcast_regression` | gotowy |
| `q_sample`: determinizm z generatorem | `test_schedule.py::test_q_sample_determinism_with_seed` | gotowy |
| loader: kształt/dtype/zakres/sortowanie/wzór resize | `test_line_dataset.py` (wszystkie) | gotowy |
| styk dataset→dyfuzor spełnia sekcję 1 | `test_contracts.py::test_styk_dataset_dyfuzor` | gotowy |
| flow end-to-end (mock, 1 krok treningu) | `test_smoke.py::test_flow_smoke` | gotowy — rośnie z pipeline: D1 realny model, D2 restore, S2 dewarp |
| degrade_g: [0,1], W, determinizm | `test_contracts.py::test_degrade_g_contract` | skip → D0 |
| dewarp: właściciel konwencji [0,1]/H=32 | `test_contracts.py::test_dewarp_contract` | skip → S2 |
| restore: przelotowość geometrii | `test_contracts.py::test_restore_passthrough_contract` | skip → D2 |
