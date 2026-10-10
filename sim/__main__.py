# SPDX-License-Identifier: MulanPSL-2.0
"""python3 -m sim [--world NAME ...] [--scene] [--png DIR]"""
from __future__ import annotations

import argparse
import logging
import os

from .world import WORLDS


def main() -> None:
    ap = argparse.ArgumentParser(prog="python3 -m sim")
    ap.add_argument("--world", action="append", choices=sorted(WORLDS),
                    help="world to run; repeat for several (default: all)")
    ap.add_argument("--scene", action="store_true",
                    help="give the controller Scene's table boxes as keep-outs")
    ap.add_argument("--strategy", action="append",
                    help="explore strategy; repeat to compare several "
                         "(default: the controller's default)")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="task timeout in simulated seconds")
    ap.add_argument("--png", metavar="DIR", help="write one PNG per run here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    from .render import render
    from .run import run

    for strategy in args.strategy or [None]:
        for name in args.world or sorted(WORLDS):
            world = WORLDS[name]()
            result = run(world, timeout_s=args.timeout, scene=args.scene,
                         strategy=strategy)
            print(f"{strategy or 'default':16s}", result.summary())
            if args.png:
                os.makedirs(args.png, exist_ok=True)
                stem = f"{name}-{strategy}" if strategy else name
                render(os.path.join(args.png, f"{stem}.png"), world,
                       result.grid, result)


if __name__ == "__main__":
    main()
