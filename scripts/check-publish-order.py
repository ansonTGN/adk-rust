#!/usr/bin/env python3
"""CI check: publish.sh order must satisfy the workspace dependency graph.

Validates, for every crate in publish.sh's CRATES list:
  1. All workspace crates are listed (and nothing extra).
  2. Every normal/build dependency on another workspace crate is published
     earlier in the sequence.
  3. Every dev-dependency that carries a version requirement is published
     earlier too — `cargo publish` resolves versioned dev-deps when generating
     the package lockfile, so an unpublished one aborts the publish (this is
     what broke the v1.0.0 release). Path-only dev-deps are stripped at
     publish and are exempt.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout)
    members = {p["name"]: p for p in meta["packages"]}

    order = []
    for line in (ROOT / "publish.sh").read_text().splitlines():
        s = line.strip()
        if re.fullmatch(r"(adk-[a-z0-9-]+|awp-types|cargo-adk)", s):
            order.append(s)
    pos = {c: i for i, c in enumerate(order)}

    errors = []
    missing = set(members) - set(order)
    extra = set(order) - set(members)
    for c in sorted(missing):
        errors.append(f"workspace crate '{c}' is missing from publish.sh")
    for c in sorted(extra):
        errors.append(f"publish.sh lists '{c}' which is not a workspace crate")

    for name, p in members.items():
        if name not in pos:
            continue
        for d in p["dependencies"]:
            if d["name"] not in pos or pos[d["name"]] <= pos[name]:
                continue
            kind = d["kind"]
            if kind in (None, "build"):
                errors.append(
                    f"{name} ({kind or 'normal'} dep) must come after {d['name']} in publish.sh"
                )
            elif kind == "dev" and d["req"] != "*":
                errors.append(
                    f"{name} dev-depends on {d['name']} {d['req']} which publishes later — "
                    f"make the dev-dep path-only or reorder"
                )

    if errors:
        print(f"check-publish-order: {len(errors)} problem(s):\n", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    print(f"check-publish-order: OK ({len(order)} crates, order fully resolvable)")


if __name__ == "__main__":
    main()
