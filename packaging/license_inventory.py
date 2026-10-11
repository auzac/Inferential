"""Generate the third-party runtime dependency inventory for LICENSES.md.

Reads installed package metadata, so run it inside the release environment:

    python packaging/license_inventory.py > packaging/third_party_inventory.md

Only runtime dependencies (those bundled into the Windows build) are listed.
"""

from __future__ import annotations

import re
from importlib import metadata

ROOT = "inferential-model-studio"


def _name(requirement: str) -> str:
    return re.split(r"[\s;<>=!~\[(]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def runtime_closure(root: str = ROOT) -> list[str]:
    seen: set[str] = set()
    stack = [root]
    while stack:
        name = stack.pop()
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for req in dist.requires or []:
            if "extra ==" in req:  # optional extras are not bundled
                continue
            dep = _name(req)
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return sorted(seen)


def license_of(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression
    classifiers = [
        c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(dict.fromkeys(classifiers))
    text = (meta.get("License") or "").strip().splitlines()
    return text[0][:80] if text else "UNKNOWN — check manually"


def main() -> None:
    print("| Package | Version | License | Home page |")
    print("|---|---|---|---|")
    for name in runtime_closure():
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            print(f"| {name} | (not installed on this platform) | | |")
            continue
        urls = dist.metadata.get_all("Project-URL") or []
        home = dist.metadata.get("Home-page") or (urls[0].split(",", 1)[-1].strip() if urls else "")
        print(f"| {dist.metadata['Name']} | {dist.version} | {license_of(dist)} | {home} |")


if __name__ == "__main__":
    main()
