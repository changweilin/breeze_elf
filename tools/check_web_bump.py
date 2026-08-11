#!/usr/bin/env python
"""Fail when a shipped ``web/`` asset changed without a cache-version bump.

The static half of this rule (index.html, the import specifiers and the service worker
all agreeing on a version) is enforced by ``tests/test_web_assets.py`` and needs no git.
This script covers the half that needs history: *did you actually bump anything*.

Without it the PWA serves a stale module against a fresh one, the named ESM import fails
at link time, and the whole controller silently never runs -- a total, invisible failure.

Usage:
    python tools/check_web_bump.py --base <ref>

Exits 0 when nothing under web/ changed, or when the versions moved as required.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

SERVICE_WORKER = "web/service-worker.js"
# Assets the browser caches by URL. A change to any of them must invalidate the cache.
CACHED_ASSETS = {
    "web/index.html",
    "web/app.js",
    "web/voice.js",
    "web/audio-utils.js",
    "web/audio-worklet.js",
    "web/manifest.webmanifest",
    SERVICE_WORKER,
}
_CACHE_NAME = re.compile(r'CACHE_NAME\s*=\s*"([^"]+)"')
_VERSIONED = re.compile(r'\./([\w.-]+\.js)\?v=(\d+)')


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def _blob(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout if result.returncode == 0 else ""


def _cache_name(text: str) -> str | None:
    match = _CACHE_NAME.search(text)
    return match.group(1) if match else None


def _versions(text: str) -> dict[str, str]:
    return dict(_VERSIONED.findall(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="ref to diff against (e.g. origin/main)")
    args = parser.parse_args()

    changed = {line for line in _git("diff", "--name-only", f"{args.base}...HEAD").split() if line}
    touched = sorted(changed & CACHED_ASSETS)
    if not touched:
        print("no cached web asset changed - nothing to check")
        return 0

    print("changed cached assets:\n  " + "\n  ".join(touched))
    problems: list[str] = []

    before_sw = _blob(args.base, SERVICE_WORKER)
    after_sw = _git("show", f"HEAD:{SERVICE_WORKER}")

    old_cache, new_cache = _cache_name(before_sw), _cache_name(after_sw)
    if old_cache and old_cache == new_cache:
        problems.append(
            f"{SERVICE_WORKER}: CACHE_NAME is still {new_cache!r} - bump it so installed "
            "PWAs discard the old precache"
        )

    # A changed module must also change its own ?v=, or importers keep the stale copy.
    old_versions, new_versions = _versions(before_sw), _versions(after_sw)
    for path in touched:
        name = path.removeprefix("web/")
        if name not in old_versions:
            continue
        if old_versions[name] == new_versions.get(name):
            problems.append(
                f"{path} changed but its ?v= is still {old_versions[name]} - bump it in "
                "index.html, in every import specifier, and in the service worker ASSETS"
            )

    if problems:
        print("\nFAIL: stale cache version\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    print(f"OK: CACHE_NAME {old_cache} -> {new_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
