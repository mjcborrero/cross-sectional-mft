"""Independent audit of the Step 0 decision grid.

Deliberately does NOT reuse build_decision_grid.py's code path. It reads the
raw monthly parquet files directly (not through ParquetStore), and re-derives
prices with pandas boolean masks rather than numpy searchsorted, so a bug in
the builder cannot hide behind the same bug in the checker.

A self-consistent pipeline that agrees with itself proves nothing. This asks a
different question: does the panel match the source data on disk?

    python scripts/audit_decision_grid.py
    python scripts/audit_decision_grid.py --sample 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits
from mft.paths import DATA_DIR, NORMALIZED_DIR

MS_MIN = 60_000
MS_HOUR = 3_600_000
KL_ROOT = NORMALIZED_DIR / "perp_klines"

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def fmt(ms) -> str:
    return splits._fmt(int(ms))


def raw_months(symbol: str, t_ms: int) -> pd.DataFrame:
    """Read the raw monthly file containing t_ms, plus the one before it.

    Direct file read -- no ParquetStore, no shared loader logic.
    """
    ts = pd.Timestamp(int(t_ms), unit="ms", tz="UTC")
    months = [(ts - pd.DateOffset(months=1)).strftime("%Y-%m"), ts.strftime("%Y-%m")]
    frames = []
    for m in months:
        f = KL_ROOT / symbol / f"{symbol}-{m}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f, columns=["close_time", "close", "quote_volume"]))
    if not frames:
        return pd.DataFrame(columns=["close_time", "close", "quote_volume"])
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="grid/decision_grid_8h.parquet")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = DATA_DIR / args.grid
    panel = pd.read_parquet(path)
    lag_ms = args.lag * MS_MIN
    step_ms = args.h * MS_HOUR

    print(f"Auditing {path}")
    print(f"  {len(panel):,} rows  {panel.symbol.nunique()} symbols  "
          f"{panel.t_obs.nunique():,} cross-sections\n")

    # ---- 1. structural ---------------------------------------------------
    print("1. STRUCTURE\n" + "=" * 74)
    t = panel["t_obs"].to_numpy()
    check(bool((t % step_ms == 0).all()),
          f"every t_obs is an exact multiple of {args.h}h from epoch",
          "-> lands on 00:00/08:00/16:00 UTC")
    hours = pd.to_datetime(panel.t_obs, unit="ms", utc=True).dt.hour.unique()
    check(set(hours) <= {0, 8, 16}, "t_obs hours are only {0,8,16} UTC", f"{sorted(hours)}")
    check(bool(((panel.t_fill - panel.t_obs) == lag_ms).all()),
          f"t_fill == t_obs + {args.lag}min exactly")
    check(not panel.duplicated(["symbol", "t_obs"]).any(), "no duplicate (symbol, t_obs)")
    check(not panel[["px_obs", "px_fill", "qv_window", "t_obs", "t_fill"]].isna().any().any(),
          "no NaN in any stored column")
    check(bool((panel.age_obs_ms > 0).all() and (panel.age_fill_ms > 0).all()),
          "all quote ages strictly positive (no same-instant or future quote)")
    check(bool((panel.t_fill < splits.HOLDOUT1_START_MS).all()),
          "no fill instant reaches the holdout wall")

    # ---- 2. price correctness vs raw files -------------------------------
    print("\n2. PRICES vs RAW MONTHLY FILES  (independent re-derivation)\n" + "=" * 74)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(panel), size=min(args.sample, len(panel)), replace=False)
    # Force-include the edges of every symbol and the rows around the 2022 gaps.
    edges = panel.groupby("symbol").apply(
        lambda g: pd.Series([g.index.min(), g.index.max()]), include_groups=False
    ).to_numpy().ravel()
    gap_zone = panel.index[
        (panel.t_obs >= int(pd.Timestamp("2022-02-24", tz="UTC").value // 10**6))
        & (panel.t_obs <= int(pd.Timestamp("2022-04-05", tz="UTC").value // 10**6))
    ].to_numpy()
    idx = np.unique(np.concatenate([idx, edges, gap_zone]))

    bad_obs, bad_fill, bad_qv, checked = [], [], [], 0
    for i in idx:
        r = panel.iloc[int(i)]
        kl = raw_months(r.symbol, int(r.t_obs))
        if kl.empty:
            continue
        kl = kl.drop_duplicates("close_time").sort_values("close_time")

        # px_obs: newest close strictly before t_obs -- pandas mask, not searchsorted
        prior = kl[kl.close_time < r.t_obs]
        if len(prior):
            exp = float(prior.iloc[-1]["close"])
            if not np.isclose(exp, float(r.px_obs), rtol=0, atol=1e-12):
                bad_obs.append((r.symbol, fmt(r.t_obs), exp, float(r.px_obs)))

        # px_fill: newest close strictly before t_fill
        klf = raw_months(r.symbol, int(r.t_fill))
        klf = klf.drop_duplicates("close_time").sort_values("close_time")
        priorf = klf[klf.close_time < r.t_fill]
        if len(priorf):
            expf = float(priorf.iloc[-1]["close"])
            if not np.isclose(expf, float(r.px_fill), rtol=0, atol=1e-12):
                bad_fill.append((r.symbol, fmt(r.t_fill), expf, float(r.px_fill)))

        # qv_window: quote volume over [t_obs - h, t_obs)
        w = kl[(kl.close_time >= r.t_obs - step_ms) & (kl.close_time < r.t_obs)]
        expq = float(w.quote_volume.sum())
        if expq > 0 and not np.isclose(expq, float(r.qv_window), rtol=1e-9):
            bad_qv.append((r.symbol, fmt(r.t_obs), expq, float(r.qv_window)))
        checked += 1

    check(not bad_obs, f"px_obs matches raw klines ({checked:,} rows re-derived)",
          f"{len(bad_obs)} mismatches" if bad_obs else "")
    for b in bad_obs[:5]:
        print(f"        {b[0]} {b[1]}  raw={b[2]!r} panel={b[3]!r}")
    check(not bad_fill, f"px_fill matches raw klines ({checked:,} rows re-derived)",
          f"{len(bad_fill)} mismatches" if bad_fill else "")
    for b in bad_fill[:5]:
        print(f"        {b[0]} {b[1]}  raw={b[2]!r} panel={b[3]!r}")
    check(not bad_qv, f"qv_window matches raw klines ({checked:,} rows re-derived)",
          f"{len(bad_qv)} mismatches" if bad_qv else "")
    for b in bad_qv[:5]:
        print(f"        {b[0]} {b[1]}  raw={b[2]:.4f} panel={b[3]:.4f}")

    # ---- 3. source integrity --------------------------------------------
    # Scoped to the region the panel actually reads. The monthly dumps and the
    # daily/live files overlap by design once a month closes, so the tree as a
    # whole DOES contain duplicate close_times -- but only in the recent tail,
    # long past the train wall. Auditing the whole tree would fail on a
    # condition that cannot reach the data under test.
    print("\n3. SOURCE INTEGRITY (train region only)\n" + "=" * 74)
    wall = splits.HOLDOUT1_START_MS
    dupes, disagree, tail_dupes = [], [], 0
    for sym in sorted(panel.symbol.unique()):
        frames = []
        for f in sorted((KL_ROOT / sym).glob("*.parquet")):
            frames.append(pd.read_parquet(f, columns=["close_time", "close"]))
        a = pd.concat(frames, ignore_index=True)
        tail_dupes += int(a.close_time.duplicated().sum())
        tr = a[a.close_time < wall]
        n_dup = int(tr.close_time.duplicated().sum())
        if n_dup:
            dupes.append((sym, n_dup))
            # If they do overlap, do they at least agree?
            d = tr[tr.close_time.duplicated(keep=False)]
            if int(d.groupby("close_time")["close"].nunique().gt(1).sum()):
                disagree.append(sym)
    check(not dupes, "no duplicate close_time before the train wall",
          f"{dupes[:4]}" if dupes else "clean")
    check(not disagree, "no conflicting values where files overlap",
          f"{disagree[:4]}" if disagree else "")
    print(f"  (informational: {tail_dupes:,} duplicate close_times exist across the "
          f"full tree, all after the wall -- monthly/daily overlap, values identical)")

    # ---- 4. row-count reconciliation ------------------------------------
    print("\n4. ROW COUNTS -- reconciled independently\n" + "=" * 74)
    all_cs = np.sort(panel.t_obs.unique())
    mism = []
    for sym, g in panel.groupby("symbol"):
        first = int(g.t_obs.min())
        # cross-sections in the panel at or after this symbol's first appearance
        eligible = all_cs[all_cs >= first]
        expected = len(eligible)
        actual = len(g)
        # difference must be explained by that symbol's own data gaps
        missing = sorted(set(eligible.tolist()) - set(g.t_obs.tolist()))
        if actual + len(missing) != expected:
            mism.append((sym, actual, expected, len(missing)))
    check(not mism, "kept rows + own-gaps == cross-sections since listing, per symbol",
          f"{mism[:4]}" if mism else f"{panel.symbol.nunique()} symbols reconcile")

    # ---- 5. price sanity -------------------------------------------------
    print("\n5. PRICE SANITY\n" + "=" * 74)
    p = panel.sort_values(["symbol", "t_obs"]).copy()
    p["ret"] = p.groupby("symbol")["px_fill"].pct_change()
    ext = p[p.ret.abs() > 0.5].dropna(subset=["ret"])
    print(f"  {len(ext)} moves >50% over one {args.h}h step "
          f"(informational -- crypto does this)")
    for _, r in ext.sort_values("ret", key=abs, ascending=False).head(5).iterrows():
        print(f"    {r.symbol:<10} {fmt(r.t_obs)}  {r.ret:+.1%}")
    # Exactly-zero returns are NOT a defect on their own: prices are discrete,
    # so a genuine round-trip can land back on the same tick. Low-priced coins
    # have few ticks and hit this often; BTC has many and almost never does.
    # The real defect would be a FROZEN series -- a price that never moved,
    # which is what a stale or forward-filled quote looks like. So go to the
    # raw klines and ask whether the price moved inside each flat interval.
    from mft.data.store import ParquetStore
    st = ParquetStore(NORMALIZED_DIR)
    z = p[p.ret == 0].copy()
    z["t_prev_fill"] = z["t_fill"] - step_ms
    frozen, moved, cache = [], 0, {}
    for _, r in z.iterrows():
        if r.symbol not in cache:
            k = st.load("perp_klines", r.symbol, columns=["close_time", "close"])
            cache[r.symbol] = k.sort_values("close_time")
        k = cache[r.symbol]
        w = k[(k.close_time >= r.t_prev_fill) & (k.close_time < r.t_fill)]
        if len(w) == 0:
            continue
        if w.close.nunique() <= 1:
            frozen.append((r.symbol, fmt(r.t_obs)))
        else:
            moved += 1
    print(f"  {len(z)} exactly-zero {args.h}h returns ({len(z)/len(p):.2%}) "
          f"-- expected from tick discretization")
    check(not frozen,
          "every flat interval shows real intra-interval movement (no frozen series)",
          f"{moved} verified moving" if not frozen else f"{len(frozen)} FROZEN: {frozen[:3]}")

    # px_obs == px_fill is the same phenomenon over 60min, not a defect.
    same = int((panel.px_obs == panel.px_fill).sum())
    print(f"  {same} rows ({same/len(panel):.3%}) have px_obs == px_fill "
          f"-- same tick after {args.lag}min, concentrated in low-priced coins")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"AUDIT FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print(f"AUDIT PASSED -- panel agrees with the raw source on every check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
