"""Rolling single-factor regression moments.

Used by Step 5 (beta window), Step 6 (idiosyncratic vol) and Step 8 (target
assembly), so it lives here rather than being re-implemented per script.

Everything is derived from rolling sums rather than per-window regressions:
exact, O(n) per column, and it returns the CENTERED moments so residual
variance can be evaluated for any beta -- not only the OLS one. That matters
because the target uses a SHRUNK beta, whose residuals are not the OLS
residuals.

All windows are backward-looking and inclusive of the current observation, so
a value stamped at t uses only data at or before t.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RollingMoments:
    """Centered rolling second moments of each column of Y against x.

    ssx  sum (x - xbar)^2        ssy  sum (y - ybar)^2
    sxy  sum (x - xbar)(y - ybar)
    n    observation count in the window
    """

    n: pd.DataFrame
    ssx: pd.DataFrame
    ssy: pd.DataFrame
    sxy: pd.DataFrame

    @property
    def beta_ols(self) -> pd.DataFrame:
        return self.sxy / self.ssx

    @property
    def var_y(self) -> pd.DataFrame:
        """Total variance of y (the denominator's ceiling)."""
        return self.ssy / (self.n - 1)

    def resid_var(self, beta: pd.DataFrame) -> pd.DataFrame:
        """Variance of (y - beta*x) for an ARBITRARY beta.

            Var(e) = Var(y) - 2b*Cov(x,y) + b^2*Var(x)

        With beta = beta_ols this reduces to the OLS residual variance and is
        guaranteed <= Var(y). With a shrunk beta it is not guaranteed -- a beta
        pushed far enough from OLS can leave MORE variance than it removes.
        That is a real property of shrinkage, not an error, and Step 6 measures
        how often it happens rather than assuming it away.
        """
        return (self.ssy - 2 * beta * self.sxy + beta * beta * self.ssx) / (self.n - 1)

    def se2_beta(self) -> pd.DataFrame:
        """Squared standard error of the OLS beta."""
        ssres = (self.ssy - self.sxy * self.sxy / self.ssx).clip(lower=0)
        return (ssres / (self.n - 2)) / self.ssx


def rolling_moments(Y: pd.DataFrame, x: pd.Series, window: int,
                    min_periods: int) -> RollingMoments:
    """Rolling centered moments of every column of Y against x.

    x is masked to each column's own availability, so every sum is taken over
    exactly the observations both series share -- otherwise a coin with gaps
    would be regressed against a market series covering different hours.
    """
    n, ssx, ssy, sxy = {}, {}, {}, {}
    for c in Y.columns:
        y = Y[c]
        xm = x.where(y.notna())
        cnt = y.rolling(window, min_periods=min_periods).count()
        sy = y.rolling(window, min_periods=min_periods).sum()
        sx = xm.rolling(window, min_periods=min_periods).sum()
        sxy_ = (xm * y).rolling(window, min_periods=min_periods).sum()
        sxx_ = (xm * xm).rolling(window, min_periods=min_periods).sum()
        syy_ = (y * y).rolling(window, min_periods=min_periods).sum()
        n[c] = cnt
        ssx[c] = sxx_ - sx * sx / cnt
        ssy[c] = syy_ - sy * sy / cnt
        sxy[c] = sxy_ - sx * sy / cnt
    return RollingMoments(pd.DataFrame(n), pd.DataFrame(ssx),
                          pd.DataFrame(ssy), pd.DataFrame(sxy))


def shrink_linear(beta_raw: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Pull each beta toward the point-in-time cross-sectional mean.

    The target is the mean of the betas observable at that instant, never a
    fixed constant and never a full-sample mean -- both would leak.
    """
    mean_t = beta_raw.mean(axis=1)
    return beta_raw.mul(lam).add(mean_t * (1.0 - lam), axis=0)
