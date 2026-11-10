#!/usr/bin/env python3
"""Pure-python re-ranking of saved hidden-test scores (no numpy/torch needed).

Usage: python3 rank.py [--base CASE] [--weights exp=w,...] [--coef exp/metric=c,...]
                       [--cases-dir DIR] [--suffix test]

Rank = sum over scored metrics of weight * |coefficient| * direction *
(value - base value) / population std of that metric over the cohort.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINED = ("reference_", "better_", "weaker_", "alt_")


def load_specs(weights: dict[str, float], coefs: dict[str, float]):
    spec = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    out = []
    for exp, setup in spec.items():
        w = weights.get(exp, setup["weight"])
        for metric, cfg in setup["metrics"].items():
            c = coefs.get(f"{exp}/{metric}", cfg["coefficient"])
            out.append((exp, metric, w * abs(c), 1.0 if cfg["higher_is_better"] else -1.0))
    return out


def flat(path: Path, method: str = "facet") -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {f"{e}/{m}": float(v) for e, p in payload["experiments"].items() for m, v in p["results"].get(method, {}).items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--weights", default="")
    ap.add_argument("--coef", default="")
    ap.add_argument("--cases-dir", default=str(ROOT / "redteam" / "hardened" / "cases"))
    ap.add_argument("--suffix", default="test")
    ap.add_argument("--json", default=None, help="write ranks to this file")
    args = ap.parse_args()
    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(",") if kv)}
    coefs = {k: float(v) for k, v in (kv.split("=") for kv in args.coef.split(",") if kv)}
    specs = load_specs(weights, coefs)
    cases = {}
    for path in sorted(Path(args.cases_dir).glob(f"*/scores_{args.suffix}.json")):
        values = flat(path)
        if values:
            cases[path.parent.name] = values
    base_name = args.base or json.loads((ROOT / "redteam" / "hardened" / "canonical.json").read_text())["case"]
    base = cases[base_name]
    trained = [c for c in cases if c.startswith(TRAINED)]
    genuine = trained + [c for c in ("simple_baseline_robust", "shortcut_passive", "shortcut_analytic_impedance", "brute_force_grid_search") if c in cases]
    cohorts = {"all": list(cases), "genuine": genuine, "trained": trained}
    scales = {}
    for name, members in cohorts.items():
        sc = {}
        for e, m, _, _ in specs:
            key = f"{e}/{m}"
            col = [cases[n][key] for n in members if key in cases[n]]
            sd = statistics.pstdev(col) if len(col) > 1 else 0.0
            sc[key] = sd if sd > 1e-12 else 1.0
        scales[name] = sc

    def rank(v, sc):
        return sum(w * d * (v[f"{e}/{m}"] - base[f"{e}/{m}"]) / sc[f"{e}/{m}"] for e, m, w, d in specs if f"{e}/{m}" in v)

    table = {c: {name: rank(v, scales[name]) for name in cohorts} for c, v in cases.items()}
    print(f"base={base_name} weights={weights or 'reference.json'} coef={coefs or 'reference.json'}")
    print(f"{'case':34s} {'all':>8s} {'genuine':>8s} {'trained':>8s}")
    for c, r in sorted(table.items(), key=lambda kv: -kv[1]["all"]):
        print(f"{c:34s} {r['all']:+8.3f} {r['genuine']:+8.3f} {r['trained']:+8.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"base": base_name, "ranks": table, "scales": scales}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
