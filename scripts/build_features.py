"""Build one feature family through the gated harness.

    python scripts/build_features.py --family carry

Families are built ONE AT A TIME, deliberately. Every family passes Stage 0
(validity, coverage, per-coin coverage), Stage 1 (ranking power) and the
truncation test before anything is written. A failure raises and writes
nothing -- gates abort, they do not print.

No family may read the target. `GridContext.target_side` asserts against it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import featbuild
from mft.featdefs.carry import Carry
from mft.featdefs.inherited import Inherited
from mft.featdefs.inherited2 import Inherited2
from mft.featdefs.liquidity import Liquidity
from mft.featdefs.path import PathAttention
from mft.featdefs.positioning import Positioning
from mft.featdefs.regime import ReturnRegime
from mft.featdefs.relational import Relational
from mft.featdefs.spotperp import SpotPerp

REGISTRY = {
    "carry": (Carry, "family_A_carry"),
    "positioning": (Positioning, "family_B_positioning"),
    "liquidity": (Liquidity, "family_C_liquidity"),
    "path": (PathAttention, "family_D_path"),
    "regime": (ReturnRegime, "family_E_regime"),
    "relational": (Relational, "family_F_relational"),
    "spotperp": (SpotPerp, "family_G_spotperp"),
    "inherited": (Inherited, "family_H_inherited"),
    "inherited2": (Inherited2, "family_I_inherited2"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--skip-truncation", action="store_true",
                    help="diagnostic only; never use for a build that is kept")
    args = ap.parse_args()

    cls, out = REGISTRY[args.family]
    try:
        featbuild.run(cls(), out, skip_truncation=args.skip_truncation)
    except featbuild.GateFailure as e:
        print("\n" + "=" * 74)
        print(f"ABORTED: {e}")
        print("Nothing written.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
