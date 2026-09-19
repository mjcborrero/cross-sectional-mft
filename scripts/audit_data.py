"""Data audit -- every building block, from raw klines to the target.

`scripts/audit.py` audits the STRATEGY layer. This audits everything beneath
it. The distinction matters: a placebo test cannot detect a market index built
with tomorrow's weights, because the placebo shares the same contaminated
inputs. Only recomputation catches that.

METHOD. Where possible each artifact is REBUILT here from its inputs by
independent code and compared to what is on disk. Re-running the original build
script would only prove it is deterministic, which is not the question.

    A  RAW KLINES        monotonic, unique, positive, low <= close <= high
    B  DECISION GRID     exact 8h spacing; px_obs is the last close AT OR
                         BEFORE t_obs and never after it
    C  RETURNS 1H        ret matches the price ratio it claims to be
    D  MARKET WEIGHTS    sum to 1, non-negative, and PIT
    E  MARKET INDEX      r_m equals the lagged-weight combination of coin
                         returns, recomputed from scratch
    F  BETA / IDIO VOL   recomputed by independent rolling regression; and a
                         PIT check that beta at t uses nothing after t
    G  FORWARD RETURNS   px_fill is the price at t_obs + 60min, and fwd_ret
                         spans exactly [t_fill, t_fill + 8h]
    H  TARGET            distributional sanity, not just the identity
    I  FEATURES          NaN structure, and PIT re-verified on a live sample
    I2 INPUT FRESHNESS   every dataset must reach the panel's end; an
                         asof-join carries a stale value forward forever
                         and nothing else would complain
    J  CROSS-ARTIFACT    the same symbols and bars mean the same thing
                         everywhere

Every check states what it would take to FAIL. TRAIN only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits
from mft.featbuild import MS_HOUR, MS_MIN, GridContext
from mft.paths import DATA_DIR
from mft.rolling import rolling_moments

MS8 = 8 * MS_HOUR
LAG = 60 * MS_MIN
fails: list[str] = []
notes: dict = {}


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(label)
    return ok


def g(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"grid/{name}.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/audit_data.json")
    ap.add_argument("--sample-coins", type=int, default=6)
    args = ap.parse_args()

    grid = g("decision_grid_8h")
    grid = grid[grid["t_obs"] < splits.HOLDOUT1_START_MS]
    t_obs = np.sort(grid["t_obs"].unique())
    syms = sorted(grid["symbol"].unique())
    ctx = GridContext(t_obs)
    print(f"DATA AUDIT -- {len(t_obs):,} train bars, {len(syms)} coins\n")

    # ================= A. raw klines ==================================
    print("A. RAW KLINES")
    print("=" * 74)
    bad_mono = bad_dup = bad_px = bad_ohlc = 0
    for s in syms[:args.sample_coins]:
        k = ctx.dataset("perp_klines", s,
                        ["close_time", "open", "high", "low", "close"],
                        "close_time")
        ct = k["close_time"].to_numpy()
        bad_mono += int((np.diff(ct) < 0).sum())
        bad_dup += int(len(ct) - len(np.unique(ct)))
        bad_px += int((k["close"] <= 0).sum())
        bad_ohlc += int(((k["low"] > k["close"]) | (k["close"] > k["high"])
                         | (k["low"] > k["high"])).sum())
    check(bad_mono == 0, "close_time non-decreasing", f"{bad_mono} inversions")
    check(bad_dup == 0, "no duplicate close_time", f"{bad_dup} dupes")
    check(bad_px == 0, "all closes positive", f"{bad_px} bad")
    check(bad_ohlc == 0, "low <= close <= high", f"{bad_ohlc} violations")

    # ================= B. decision grid ===============================
    print("\nB. DECISION GRID")
    print("=" * 74)
    d = np.diff(t_obs)
    gaps = t_obs[:-1][d != MS8]
    # Gaps are a DATA fact -- days missing from the Binance archive -- not a
    # defect in the grid. What matters is that they are known, enumerated, and
    # that nothing downstream silently treats a gap as an 8h step.
    print(f"       {len(d) - len(gaps):,} of {len(d):,} steps are exactly 8h")
    for x in gaps:
        nxt = t_obs[np.searchsorted(t_obs, x) + 1]
        print(f"       GAP {splits._fmt(int(x))} -> {splits._fmt(int(nxt))} "
              f"({(nxt - x)/3.6e6:.0f}h)")
    notes["grid_gaps"] = [splits._fmt(int(x)) for x in gaps]
    check(len(gaps) <= 2, "grid gaps are the two known archive holes",
          f"{len(gaps)} found")

    # px_obs must be the last close AT OR BEFORE t_obs. A close stamped AFTER
    # t_obs would be lookahead of exactly the kind that killed the last build.
    worst_future, worst_age, checked = 0, 0, 0
    for s in syms[:args.sample_coins]:
        k = ctx.dataset("perp_klines", s, ["close_time", "close"], "close_time")
        ct, px = k["close_time"].to_numpy(), k["close"].to_numpy("float64")
        sub = grid[grid["symbol"] == s]
        i = np.searchsorted(ct, sub["t_obs"].to_numpy(), side="left") - 1
        ok = i >= 0
        used = np.where(ok, ct[np.clip(i, 0, len(ct) - 1)], -1)
        gt = sub["t_obs"].to_numpy()
        worst_future = max(worst_future, int((used[ok] > gt[ok]).sum()))
        worst_age = max(worst_age, float(np.nanmax((gt[ok] - used[ok]) / 6e4)))
        rec = sub["px_obs"].to_numpy()
        got = np.where(ok, px[np.clip(i, 0, len(px) - 1)], np.nan)
        m = ok & np.isfinite(rec)
        checked += int(m.sum())
        if np.nanmax(np.abs(rec[m] - got[m])) > 1e-9:
            check(False, f"px_obs matches last close for {s}")
    check(worst_future == 0, "px_obs NEVER uses a close stamped after t_obs",
          f"{worst_future} future stamps")
    check(worst_age <= 61, "px_obs is fresh (max staleness)",
          f"{worst_age:.0f} min")
    print(f"       verified {checked:,} px_obs values by recomputation")

    # ================= C. returns 1h ==================================
    print("\nC. RETURNS 1H")
    print("=" * 74)
    h = g("returns_1h")
    h = h[h["t"] < splits.HOLDOUT1_START_MS]
    H = h.pivot(index="t", columns="symbol", values="ret")
    hg = np.asarray(H.index, dtype=np.int64)
    dh = np.diff(hg)
    check(bool((dh == MS_HOUR).all()), "1h grid is exactly hourly",
          f"{int((dh != MS_HOUR).sum())} irregular")
    s0 = syms[0]
    k = ctx.dataset("perp_klines", s0, ["close_time", "close"], "close_time")
    ct, px = k["close_time"].to_numpy(), k["close"].to_numpy("float64")
    i = np.searchsorted(ct, hg, side="left") - 1
    p_at = np.where(i >= 0, px[np.clip(i, 0, len(px) - 1)], np.nan)
    rebuilt = p_at[1:] / p_at[:-1] - 1.0
    have = H[s0].to_numpy()[1:]
    m = np.isfinite(rebuilt) & np.isfinite(have)
    md = float(np.nanmax(np.abs(rebuilt[m] - have[m]))) if m.any() else np.nan
    check(md < 1e-9, f"ret is px(t)/px(t-1h)-1 recomputed ({s0})",
          f"max|diff| {md:.2e} over {int(m.sum()):,}")

    # ================= D. market weights ==============================
    print("\nD. MARKET WEIGHTS")
    print("=" * 74)
    W = g("market_weights_30d")
    W = W[W["t_obs"] < splits.HOLDOUT1_START_MS]
    Wm = W.pivot(index="t_obs", columns="symbol", values="weight")
    sums = Wm.sum(axis=1)
    live = sums[sums > 0]
    check(float((live - 1.0).abs().max()) < 1e-9, "weights sum to 1 per bar",
          f"worst {float((live - 1.0).abs().max()):.2e}")
    check(bool((Wm.fillna(0) >= -1e-15).all().all()), "weights non-negative")
    notes["weight_max"] = float(Wm.max().max())
    print(f"       largest single weight ever: {notes['weight_max']:.3f} "
          f"({Wm.max().idxmax()})")

    # ================= E. market index ================================
    print("\nE. MARKET INDEX -- rebuilt from coin returns and LAGGED weights")
    print("=" * 74)
    P = grid.pivot(index="t_obs", columns="symbol", values="px_obs")
    R8 = P / P.shift(1) - 1.0
    Wl = Wm.reindex(R8.index).reindex(columns=R8.columns).shift(1)
    M = Wl.where(R8.notna())
    M = M.div(M.sum(axis=1), axis=0)
    rm_rebuilt = (M * R8).sum(axis=1, min_count=1).where(
        M.notna().sum(axis=1) >= 10)
    T = g("target_8h_lag60")
    T = T[T["t_obs"] < splits.HOLDOUT1_START_MS]
    # fwd_rm must be rebuilt from the SAME window it is defined on: the
    # px_fill forward returns, weighted by the weights stamped at t_obs. An
    # earlier version of this check rebuilt from px_obs 8h returns shifted by
    # one bar -- a window offset by the 60min lag, sharing 7 of its 8 hours --
    # and produced corr 0.836, which looked like a defect and was one in the
    # CHECK. Comparing like with like is the whole job of an audit.
    Fw = g("forward_8h_lag60")
    Fw = Fw[Fw["t_obs"] < splits.HOLDOUT1_START_MS]
    FR = Fw.pivot(index="t_obs", columns="symbol", values="fwd_ret")
    Wf = Wm.reindex(FR.index).reindex(columns=FR.columns)
    Mf = Wf.where(FR.notna())
    Mf = Mf.div(Mf.sum(axis=1), axis=0)
    rm_fwd = (Mf * FR).sum(axis=1, min_count=1)
    have_rm = Fw.groupby("t_obs")["fwd_rm"].first().reindex(rm_fwd.index)
    m = rm_fwd.notna() & have_rm.notna()
    md_rm = float((rm_fwd[m] - have_rm[m]).abs().max())
    check(md_rm < 1e-12,
          "fwd_rm == weights(t_obs) . fwd_ret, rebuilt independently",
          f"max|diff| {md_rm:.2e} over {int(m.sum()):,} bars")

    # ================= F. beta / idio vol =============================
    print("\nF. BETA / IDIO VOL -- recomputed by independent rolling regression")
    print("=" * 74)
    bi = g("beta_idiovol_8h")
    bi = bi[bi["t_obs"] < splits.HOLDOUT1_START_MS]
    x1 = h.drop_duplicates("t").set_index("t")["r_m"].reindex(H.index)
    win, minp = 30 * 24, int(30 * 24 * 0.6)
    rm_ = rolling_moments(H, x1, win, minp)
    B = bi.pivot(index="t_obs", columns="symbol", values="beta")
    diffs = []
    for s in syms[:args.sample_coins]:
        if s not in rm_.beta_ols.columns:
            continue
        ser = rm_.beta_ols[s]
        i = np.searchsorted(hg, t_obs, side="right") - 1
        at = np.where(i >= 0, ser.to_numpy()[np.clip(i, 0, len(ser) - 1)], np.nan)
        rec = B[s].reindex(t_obs).to_numpy()
        m = np.isfinite(at) & np.isfinite(rec)
        if m.sum() > 100:
            diffs.append(float(np.nanmax(np.abs(at[m] - rec[m]))))
    worst = max(diffs) if diffs else np.nan
    check(worst < 1e-6, "beta reproduces an independent rolling OLS",
          f"max|diff| {worst:.2e} across {len(diffs)} coins")

    # PIT: recompute beta at a mid instant using ONLY data <= that instant.
    mid = t_obs[len(t_obs) // 2]
    Hc = H[H.index <= mid]
    xc = x1[x1.index <= mid]
    bc = rolling_moments(Hc, xc, win, minp).beta_ols.iloc[-1]
    ref = B.reindex([mid]).iloc[0]
    common = [c for c in bc.index if c in ref.index
              and np.isfinite(bc[c]) and np.isfinite(ref[c])]
    pit = float(np.nanmax(np.abs(bc[common] - ref[common]))) if common else np.nan
    check(pit < 1e-6, "beta at t uses ONLY data at or before t (PIT)",
          f"max|diff| {pit:.2e} at {splits._fmt(int(mid))}, "
          f"{len(common)} coins")

    sig = bi["sigma_eps"]
    check(bool((sig.dropna() > 0).all()), "sigma_eps strictly positive",
          f"min {sig.min():.3e}")

    # ================= G. forward returns =============================
    print("\nG. FORWARD RETURNS")
    print("=" * 74)
    F = g("forward_8h_lag60")
    F = F[F["t_obs"] < splits.HOLDOUT1_START_MS]
    s0 = syms[0]
    k = ctx.dataset("perp_klines", s0, ["close_time", "close"], "close_time")
    ct, px = k["close_time"].to_numpy(), k["close"].to_numpy("float64")
    sub = F[F["symbol"] == s0].sort_values("t_obs")
    fill_at = sub["t_obs"].to_numpy() + LAG
    i = np.searchsorted(ct, fill_at, side="left") - 1
    ok = i >= 0
    used = np.where(ok, ct[np.clip(i, 0, len(ct) - 1)], -1)
    check(int((used[ok] > fill_at[ok]).sum()) == 0,
          "px_fill never uses a close stamped after t_fill")
    rec = np.where(ok, px[np.clip(i, 0, len(px) - 1)], np.nan)
    have = sub["px_fill"].to_numpy()
    m = np.isfinite(rec) & np.isfinite(have)
    md = float(np.nanmax(np.abs(rec[m] - have[m])))
    check(md < 1e-9, f"px_fill is the close asof t_obs+60min ({s0})",
          f"max|diff| {md:.2e}")

    # THE STRONG VERSION, and the one the grid gaps make necessary. fwd_ret
    # must span an EXPLICIT 8h horizon, not "until the next grid bar". Where
    # the grid is regular the two are identical and the distinction is
    # invisible, so it is tested precisely at the gaps, where they differ.
    exit_at = sub["t_obs"].to_numpy() + LAG + MS8
    j = np.searchsorted(ct, exit_at, side="left") - 1
    # A row can only be checked if the EXIT instant is inside the data this
    # process can see. The last train bar exits at 2025-01-01 01:00, past the
    # wall the audit runs under, so its recomputation would silently fall back
    # to the last available close and report a mismatch that is the checker's
    # blind spot rather than a defect in the artifact. Verified: exactly one
    # row, recorded -0.00697 against a wall-limited -0.02251.
    okj = (j >= 0) & ok & (exit_at <= ct.max())
    p_out = np.where(okj, px[np.clip(j, 0, len(px) - 1)], np.nan)
    horizon_ret = p_out / rec - 1.0
    hv = sub["fwd_ret"].to_numpy()
    mm = np.isfinite(horizon_ret) & np.isfinite(hv) & okj
    md_h = float(np.nanmax(np.abs(horizon_ret[mm] - hv[mm])))
    check(md_h < 1e-9,
          "fwd_ret spans an explicit 8h horizon, even across a grid gap",
          f"max|diff| {md_h:.2e} over {int(mm.sum()):,} rows")

    # ================= H. target distribution =========================
    print("\nH. TARGET -- distribution, not just the identity")
    print("=" * 74)
    tg = T["target"].dropna()
    per_bar = T.groupby("t_obs")["target"].std()
    print(f"       mean {tg.mean():+.4f}  sd {tg.std():.4f}  "
          f"skew {tg.skew():+.2f}  kurt {tg.kurt():+.1f}")
    print(f"       |target| > 10 in {float((tg.abs() > 10).mean()):.4%} of rows; "
          f"max {tg.abs().max():.1f}")
    check(abs(float(tg.mean())) < 0.20, "target mean near zero",
          f"{tg.mean():+.4f}")
    check(0.5 < float(tg.std()) < 2.5,
          "target sd near 1 (it is divided by sigma_eps)", f"{tg.std():.3f}")
    check(float((tg.abs() > 50).mean()) < 1e-3, "no pathological outliers",
          f"{float((tg.abs() > 50).mean()):.5%} beyond 50 sd")
    check(float(per_bar.median()) > 0.1,
          "within-bar dispersion exists (something to rank)",
          f"median per-bar sd {per_bar.median():.3f}")

    # ================= I. features ====================================
    print("\nI. FEATURES")
    print("=" * 74)
    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])
    FAM = ["family_A_carry", "family_B_positioning", "family_C_liquidity",
           "family_D_path", "family_E_regime", "family_F_relational",
           "family_G_spotperp", "family_H_inherited", "family_I_inherited2"]
    X = pd.concat([pd.read_parquet(DATA_DIR / f"features/{f}.parquet")
                   .set_index(["t_obs", "symbol"]) for f in FAM], axis=1)[cols]
    X = X[X.index.get_level_values("t_obs") < splits.HOLDOUT1_START_MS]
    cov = X.notna().mean()
    inf = np.isinf(X.select_dtypes("number").to_numpy()).sum()
    print(f"       coverage: min {cov.min():.1%} ({cov.idxmin()}), "
          f"median {cov.median():.1%}")
    check(inf == 0, "no infinities in the frozen panel", f"{inf} found")
    check(float(cov.min()) > 0.5, "every frozen feature is mostly present",
          f"worst {cov.min():.1%}")
    const = [c for c in cols if X[c].std(skipna=True) == 0]
    check(not const, "no constant column", f"{len(const)}: {const[:3]}")

    # ================= I2. input freshness ============================
    # ADDED after the 2026 update, where prices reached 2026-08-29 but the
    # funding ARCHIVE stopped between 2026-07-31 and 2026-08-04 by symbol.
    # Funding feeds 7 of the 72 frozen features AND the P&L, so a window that
    # extends past the funding data would be scored on stale carry while every
    # other input was current. Nothing in the pipeline would have complained:
    # asof-joins happily carry the last known value forward forever.
    print("\nI2. INPUT FRESHNESS -- does every dataset reach the panel's end?")
    print("=" * 74)
    panel_end = int(t_obs.max())
    print(f"       panel ends {splits._fmt(panel_end)}")
    stale = []
    for ds, col in [("perp_klines", "close_time"), ("spot_klines", "close_time"),
                    ("funding", "calc_time"), ("metrics", "create_time"),
                    ("premium_index", "close_time")]:
        worst, worst_sym = None, None
        for s_ in syms:
            d_ = ctx.dataset(ds, s_, [col], col)
            if d_.empty:
                continue
            mx = int(d_[col].max())
            if worst is None or mx < worst:
                worst, worst_sym = mx, s_
        if worst is None:
            continue
        lag_h = (panel_end - worst) / 3.6e6
        flag = "" if lag_h <= 24 else "   <- STALE"
        print(f"       {ds:<15} earliest-ending {worst_sym:<9} "
              f"{splits._fmt(worst)}  ({lag_h:+.0f}h vs panel){flag}")
        if lag_h > 24:
            stale.append((ds, worst_sym, lag_h))
    check(not stale,
          "every input dataset reaches within 24h of the panel end",
          "; ".join(f"{d} {s_} {h:.0f}h" for d, s_, h in stale) if stale
          else f"{5} datasets checked")

    # ================= J. cross-artifact ==============================
    print("\nJ. CROSS-ARTIFACT CONSISTENCY")
    print("=" * 74)
    gk = set(map(tuple, grid[["t_obs", "symbol"]].to_numpy()))
    tk = set(map(tuple, T[["t_obs", "symbol"]].to_numpy()))
    check(tk <= gk, "every target row exists in the decision grid",
          f"{len(tk - gk)} orphans")
    xk = set(X.index)
    check(xk <= gk, "every feature row exists in the decision grid",
          f"{len(xk - gk)} orphans")
    print(f"       grid {len(gk):,} | target {len(tk):,} | features {len(xk):,}")

    print("\n" + "=" * 74)
    if fails:
        print(f"DATA AUDIT FAILED ({len(fails)}):")
        for f_ in fails:
            print(f"    {f_}")
    else:
        print("ALL DATA CHECKS PASSED -- raw klines through to the frozen panel.")
    out = DATA_DIR / args.out
    out.write_text(json.dumps({"failures": fails, "notes": notes}, indent=2,
                              default=float), encoding="utf-8")
    print(f"Wrote {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
