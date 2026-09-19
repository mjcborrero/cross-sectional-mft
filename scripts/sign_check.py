"""Sign check against pre-registered priors. FEATURE_SELECTION.md §7 item 3.

"A feature whose realised IC is OPPOSITE to its pre-registered sign is evidence
of noise, not of a discovery -- and it must be reported as such rather than
reinterpreted with a new story."

WHY THIS IS WORTH RUNNING, AND WHY IT IS ALMOST FREE
----------------------------------------------------
It spends no additional target contact: the ICs already exist, measured once by
Stage 6. All this does is compare their signs to predictions that were written
down before any of them were seen. There is no way to tune it.

It answers a question no importance ranking can. An importance score says a
feature is being used; it cannot say whether the reason the designer gave for
building it was right. If the priors carry no information -- if matches land at
50% -- then the design process was decorative and the surviving features are
better understood as a data-mined set, with everything that implies for how
much the eventual out-of-sample result should be believed.

WHO IS ELIGIBLE
---------------
Only DESIGNED features, families A-G, which FEATURE_LIST_FROZEN.md §3 describes
as "predictions". Families H and I are the inherited library and §4 states
plainly that they are "**not** hypotheses -- no individual expected sign". They
are excluded rather than assigned a sign after the fact.

Four designed features are marked `**?**` in the source table -- beta_momentum,
sector_cohesion, sector_belonging, sector_decoupling. A declared absence of
prior is honest and is respected: they are excluded too, not silently counted
as whichever way they landed.

THE TEST
--------
Among features whose IC is distinguishable from zero, the share matching their
prior is compared against the 50% a coin would give, via a two-sided binomial
test. Signs are read from the frozen list's own tables, so the priors cannot
drift between the document and the check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft.paths import DATA_DIR

DOC = Path(__file__).resolve().parent.parent / "docs/FEATURE_LIST_FROZEN.md"
T_SIGNIFICANT = 2.0

# U+2212 MINUS SIGN and ASCII hyphen both appear in the tables.
_NEG = {"−", "-"}
_POS = {"+"}


def parse_priors(path: Path) -> dict[str, dict]:
    """Read expected signs straight from the frozen list's tables.

    Parsed rather than transcribed so the prior in the document and the prior
    in this check cannot diverge.
    """
    txt = path.read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*([A-G]\d+[a-z]?)\s*\|\s*`([^`]+)`\s*\|(.*)$",
                      txt, re.M)
    out = {}
    for code, name, rest in rows:
        m = re.search(r"\*\*([^*]+)\*\*", rest)
        raw = m.group(1).strip() if m else "?"
        sign = +1 if raw in _POS else (-1 if raw in _NEG else 0)
        conf = rest.rstrip().rstrip("|").split("|")[-1].strip()
        out[name] = {"code": code, "sign": sign, "raw": raw, "confidence": conf}
    return out


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value, by total probability of outcomes at
    most as likely as the observed one."""
    if n == 0:
        return 1.0
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(q for q in probs if q <= obs * (1 + 1e-12))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/sign_check.json")
    args = ap.parse_args()

    priors = parse_priors(DOC)
    s6 = json.loads((DATA_DIR / "features/stage6_univariate.json").read_text())
    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())

    rows = []
    for name, meta in fz["features"].items():
        r = s6["results"].get(name)
        if r is None or r["ic"] is None:
            continue
        p = priors.get(name)
        if p is None:               # inherited library: not a hypothesis
            continue
        rows.append({"feature": name, "code": p["code"], "family": meta["family"],
                     "prior": p["sign"], "prior_raw": p["raw"],
                     "confidence": p["confidence"],
                     "ic": r["ic"], "t": r["t"],
                     "folds_same_sign": r["folds_same_sign"]})
    R = pd.DataFrame(rows).set_index("feature")

    print("SIGN CHECK vs PRE-REGISTERED PRIORS")
    print(f"  priors parsed from {DOC.name}: {len(priors)} designed features")
    print(f"  matched to Stage 6 results: {len(R)}")

    nodir = R[R["prior"] == 0]
    R = R[R["prior"] != 0].copy()
    print(f"  excluded, prior declared '?' ({len(nodir)}): "
          + ", ".join(nodir.index))
    print(f"  excluded, inherited library (H/I): not hypotheses, no sign\n")

    R["match"] = (R["ic"] > 0).astype(int) * 2 - 1 == R["prior"]
    sig = R[R["t"].abs() >= T_SIGNIFICANT]

    print(f"ALL DESIGNED FEATURES WITH A DIRECTIONAL PRIOR ({len(R)})")
    print("=" * 74)
    k, n = int(R["match"].sum()), len(R)
    print(f"  sign matches prior: {k}/{n} = {k/n:.1%}   "
          f"binomial p {binom_two_sided(k, n):.3f}")

    print(f"\nRESTRICTED TO |t| >= {T_SIGNIFICANT} ({len(sig)}) -- the only ones "
          f"where the\nsign is actually measured rather than a coin flip")
    print("=" * 74)
    ks, ns = int(sig["match"].sum()), len(sig)
    pval = binom_two_sided(ks, ns)
    print(f"  sign matches prior: {ks}/{ns} = {ks/ns:.1%}   binomial p {pval:.4f}")

    print(f"\n  {'feature':<30} {'code':<5} {'prior':>5} {'IC':>8} {'t':>7}  ok")
    for c in sig.sort_values("t", key=lambda s: s.abs(), ascending=False).index:
        r = sig.loc[c]
        print(f"  {c:<30} {r['code']:<5} {r['prior_raw']:>5} {r['ic']:>+8.4f} "
              f"{r['t']:>+7.2f}  {'yes' if r['match'] else 'NO'}")

    wrong = sig[~sig["match"]]
    print("\n" + "=" * 74)
    print(f"SIGNIFICANT AND WRONG-SIGNED ({len(wrong)})")
    if len(wrong):
        print("  Per §7 item 3 these are evidence of NOISE, not discoveries with")
        print("  a new story. Recorded, and NOT reinterpreted:")
        for c in wrong.index:
            print(f"    {c:<30} prior {wrong.loc[c,'prior_raw']}, "
                  f"IC {wrong.loc[c,'ic']:+.4f}, t {wrong.loc[c,'t']:+.2f}, "
                  f"conf {wrong.loc[c,'confidence']}")
    else:
        print("  none")

    print("\nINTERPRETATION")
    print("=" * 74)
    if pval < 0.05 and ks / ns > 0.5:
        print(f"  The priors carry information ({ks/ns:.0%} vs 50%, p {pval:.4f}).")
        print("  The design process was doing work, not decorating a search.")
    else:
        print(f"  The priors are INDISTINGUISHABLE FROM A COIN FLIP: "
              f"{ks}/{ns} = {ks/ns:.0%}, p {pval:.4f}.")
        print("  Being above 50% is not the same as being better than chance,")
        print("  and at n={} it is not evidence of anything.".format(ns))
        print()
        print("  This does NOT say the features fail to predict -- 28 of 69 reach")
        print("  |t| >= 2 where chance gives ~3, so something is there. It says")
        print("  the ECONOMIC REASONING did not predict the direction. The")
        print("  surviving set must therefore be treated as DATA-MINED rather")
        print("  than as confirmed hypotheses, and the out-of-sample result")
        print("  discounted accordingly. Claiming otherwise would be exactly the")
        print("  retrofitting §7 item 3 was written to prevent.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "check": "sign_vs_prior", "t_significant": T_SIGNIFICANT,
        "n_with_prior": int(n), "n_match": int(k),
        "n_significant": int(ns), "n_significant_match": int(ks),
        "binomial_p_significant": pval,
        "excluded_no_direction": sorted(nodir.index),
        "wrong_signed": {c: {"prior": wrong.loc[c, "prior_raw"],
                             "ic": float(wrong.loc[c, "ic"]),
                             "t": float(wrong.loc[c, "t"])} for c in wrong.index},
        "per_feature": {c: {"code": R.loc[c, "code"],
                            "prior": int(R.loc[c, "prior"]),
                            "ic": float(R.loc[c, "ic"]),
                            "t": float(R.loc[c, "t"]),
                            "match": bool(R.loc[c, "match"])} for c in R.index},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
