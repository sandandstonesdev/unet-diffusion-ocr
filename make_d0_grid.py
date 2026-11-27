"""Siatka degradacji G — bramka gotowości D0 (README, kamienie milowe).

Definicja gotowości D0: 200 próbek siatki t_degr = 20 linii val (deterministyczny
wybór po seedzie z configs/diffusion.yaml) x 10 poziomów t_frac (t_degr_grid +
t=0.0 jako odniesienie). Artifacts w runs/d0_grid/ (IGNOROWANE przez git):

- samples/{stem}_t{t_frac}.png — zdegradowane linie (uint8, jak z [0,1]),
- manifest.jsonl — {line, out, t_frac, seed, sha256}: ten sam (line, t_frac,
  seed) odtwarza próbkę bajtowo; decyzje D1+ opierają się na manifeście,
  nie na obrazkach z notebooka,
- contact_sheet.png — arkusz 20x10 do oceny wzrokiem (notatka, nie bramka).

Uruchomienie: python make_d0_grid.py               (val z configs/diffusion.yaml)
              python make_d0_grid.py --synthetic   (linie PIL, gdy val puste)
              python make_d0_grid.py --limit 3     (szybki test mechaniki)
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")           # polskie znaki w printach (Windows)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from io import BytesIO  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")                              # headless: bez okna wykresu

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from configs.schedule import cosine_schedule  # noqa: E402
from data.degrade import degrade_g  # noqa: E402
from data.line_dataset import LineDataset  # noqa: E402

SYNTH_TEXTS = [                                    # jak notebooks/d0_schedule_inspect
    "Zażółć gęślą jaźń", "Sed ut perspiciatis", "Pchnąć w tę łódź jeża",
    "Strzeżona zagroda", "Mężył dżdżysty krzew",
]


def _synthetic_line(text: str, w: int) -> torch.Tensor:
    """Linia PIL-em -> (1,32,W) [0,1] (jak testy/notebooki, gdy data/raw puste)."""
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    im = Image.new("L", (w, 32), 255)
    ImageDraw.Draw(im).text((6, 2), text, fill=30, font=font)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _pick_lines(val_dir: Path, n_lines: int, seed: int, synthetic: bool):
    """(path, (1,32,W)) x n_lines — deterministycznie: sorted + Random(seed).sample."""
    if synthetic:
        widths = [192 + 8 * i for i in range(n_lines)]
        texts = [SYNTH_TEXTS[i % len(SYNTH_TEXTS)] for i in range(n_lines)]
        return [(f"synthetic:{txt}", _synthetic_line(txt, w))
                for txt, w in zip(texts, widths, strict=True)]

    ds = LineDataset(str(val_dir))
    if len(ds) == 0:
        raise SystemExit(
            f"Brak PNG w {val_dir} — pobierz ocr-pl-lines val (README, sekcja Dane) "
            "albo uruchom z --synthetic.")
    if len(ds) < n_lines:
        print(f"UWAGA: val ma {len(ds)} linii (< {n_lines}) — używam wszystkich.")
    idx = sorted(random.Random(seed).sample(range(len(ds)), min(n_lines, len(ds))))
    return [(str(ds.files[i]), ds[i]["image"]) for i in idx]


def _save_png(deg: torch.Tensor, path: Path) -> str:
    """(1,32,W) [0,1] -> PNG uint8; zwraca sha256 zapisanych bajtów."""
    arr = (deg[0] * 255).round().clamp(0, 255).byte().numpy()
    Image.fromarray(arr, mode="L").save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Siatka degradacji G — bramka D0")
    ap.add_argument("--config", default="configs/diffusion.yaml")
    ap.add_argument("--limit", type=int, default=0,
                    help="skróć liczbę linii (szybki test mechaniki)")
    ap.add_argument("--synthetic", action="store_true",
                    help="pomiń data/raw/val — linie syntetyczne PIL")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    dif, grid = cfg["diffusion"], cfg["grid"]
    sched = cosine_schedule(int(dif["T"]), s=float(dif["s"]),
                            beta_max=float(dif["beta_max"]))
    levels = [0.0] + [float(t) for t in dif["t_degr_grid"]]    # t=0: odniesienie
    n_lines = min(int(grid["lines_for_d0"]), args.limit) if args.limit \
        else int(grid["lines_for_d0"])
    out_dir = Path(grid["out"])
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    lines = _pick_lines(Path(cfg["paths"]["raw_val"]), n_lines, seed,
                        args.synthetic)
    source = "SYNTETYCZNE (--synthetic)" if args.synthetic \
        else f"{cfg['paths']['raw_val']} (ocr-pl-lines val)"

    # siatka: batch = 1 na próbke (W zmienna per linia — bez paddingu, kontrakt
    # sekcji 1); seed per próbka -> manifest odtwarza każdą próbkę niezależnie
    manifest_path = out_dir / "manifest.jsonl"
    rows, cells = [], {}
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for r, (path, img) in enumerate(lines):                 # (1,32,W) [0,1]
            for c, tf in enumerate(levels):
                smp_seed = seed + r * len(levels) + c
                deg = degrade_g(img, t_frac=tf, schedule=sched, seed=smp_seed)
                stem = Path(path).stem if not path.startswith("synthetic:") \
                    else f"synth{r:02d}"
                out_png = samples_dir / f"{stem}_t{tf:.1f}.png"
                sha = _save_png(deg, out_png)
                row = {"line": path, "out": str(out_png), "t_frac": tf,
                       "seed": smp_seed, "sha256": sha}
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                cells[(r, c)] = deg[0].numpy()

    # sanity: manifest odtwarza próbkę (ta sama linia + ten sam seed -> ten sam PNG)
    e0 = rows[0]
    img0 = lines[0][1]
    re = degrade_g(img0, t_frac=e0["t_frac"], schedule=sched, seed=e0["seed"])
    buf = BytesIO()
    Image.fromarray((re[0] * 255).round().clamp(0, 255).byte().numpy(),
                    mode="L").save(buf, format="PNG")
    assert hashlib.sha256(buf.getvalue()).hexdigest() == e0["sha256"], \
        "manifest nie odtwarza próbki"

    # contact-sheet (wiersz = linia, kolumna = t_frac) — jak notebook D0
    fig, axes = plt.subplots(len(lines), len(levels),
                             figsize=(1.6 * len(levels), 1.1 * len(lines)))
    for (r, c), img in cells.items():
        ax = axes[r, c] if len(lines) > 1 else axes[c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_axis_off()
        if r == 0:
            ax.set_title(f"t={levels[c]}", fontsize=7)
    fig.suptitle(f"D0 — siatka G: wiersz = linia, kolumna = t_frac ({source})",
                 y=1.02, fontsize=9)
    fig.tight_layout()
    sheet = out_dir / "contact_sheet.png"
    fig.savefig(sheet, dpi=140, bbox_inches="tight")
    plt.close(fig)

    n, unique = len(rows), len({r["sha256"] for r in rows})
    target = int(dif["d0_samples"])
    print(f"linie: {len(lines)} x poziomy: {levels} ({source})")
    print(f"próbki: {n} (cel D0: {target})  unikalne sha256: {unique}")
    print(f"manifest: {manifest_path}")
    print(f"sha256(manifest) = "
          f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}")
    print(f"contact-sheet: {sheet}")
    if n != target:
        print(f"UWAGA: {n} != {target} — bramka D0 NIEspełniona")
    elif unique != n:
        print("UWAGA: duplikaty sha w siatce — sprawdź seedy/siatkę")
    else:
        print("OK — bramka D0 spełniona: 200 próbek, manifest odtwarza próbki")


if __name__ == "__main__":
    main()
