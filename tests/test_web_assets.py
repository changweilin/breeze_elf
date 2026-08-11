"""Static invariants for the shipped ``web/`` bundle and for the test suite itself.

These are the two rules that have broken silently in the past and that no other test
covers:

1. **Cache versions must agree.** ``index.html`` loads ``app.js``/``voice.js`` with a
   ``?v=`` query, ``voice.js`` imports ``audio-utils.js`` with one, and
   ``service-worker.js`` precaches all of them by URL. If any of those drift apart, a
   phone loads a fresh module against a stale dependency, the named ESM import fails at
   link time, and the whole controller silently never executes.
2. **Tests must be ``unittest.TestCase``.** CI runs ``unittest discover``, which collects
   nothing from pytest-style module-level functions -- such a file looks green while
   testing nothing at all.
"""

import re
import unittest
from pathlib import Path

from breeze_elf import main

WEB_DIR = Path(main.WEB_DIR)
TESTS_DIR = Path(__file__).resolve().parent

_SCRIPT_TAG = re.compile(r'<script\s+src="\./(?P<file>[\w.-]+\.js)\?v=(?P<version>\d+)"')
_IMPORT_SPEC = re.compile(r'from\s+"\./(?P<file>[\w.-]+\.js)\?v=(?P<version>\d+)"')
_SW_ASSET = re.compile(r'"\./(?P<path>[^"]*)"')
_CACHE_NAME = re.compile(r'CACHE_NAME\s*=\s*"(?P<name>[^"]+)"')


def _read(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def _versioned_urls() -> dict[str, str]:
    """Every ``<file>.js?v=N`` referenced by the shell or by another module."""
    urls: dict[str, str] = {}
    sources = [_read("index.html")] + [
        path.read_text(encoding="utf-8") for path in sorted(WEB_DIR.glob("*.js"))
    ]
    for text in sources:
        for pattern in (_SCRIPT_TAG, _IMPORT_SPEC):
            for match in pattern.finditer(text):
                urls[match.group("file")] = match.group("version")
    return urls


class CacheVersionTests(unittest.TestCase):
    def setUp(self):
        self.service_worker = _read("service-worker.js")
        self.assets = set(_SW_ASSET.findall(self.service_worker))

    def test_every_versioned_url_is_precached(self):
        missing = [
            f"{file}?v={version}"
            for file, version in _versioned_urls().items()
            if f"{file}?v={version}" not in self.assets
        ]
        self.assertEqual(
            missing,
            [],
            "service-worker.js ASSETS is stale: bump these URLs (and CACHE_NAME) together "
            "with index.html / the import specifier",
        )

    def test_one_version_per_file(self):
        """index.html and the importing module must not disagree about a file."""
        seen: dict[str, set[str]] = {}
        for text in [_read("index.html")] + [
            path.read_text(encoding="utf-8") for path in sorted(WEB_DIR.glob("*.js"))
        ]:
            for pattern in (_SCRIPT_TAG, _IMPORT_SPEC):
                for match in pattern.finditer(text):
                    seen.setdefault(match.group("file"), set()).add(match.group("version"))
        conflicts = {file: sorted(v) for file, v in seen.items() if len(v) > 1}
        self.assertEqual(conflicts, {}, "same file referenced at two different ?v= versions")

    def test_precached_assets_exist_on_disk(self):
        for asset in sorted(self.assets):
            if not asset or asset.startswith("http"):
                continue
            name = asset.split("?", 1)[0]
            self.assertTrue((WEB_DIR / name).is_file(), f"precached but missing: {asset}")

    def test_cache_name_is_present_and_versioned(self):
        match = _CACHE_NAME.search(self.service_worker)
        self.assertIsNotNone(match, "service-worker.js must define CACHE_NAME")
        self.assertRegex(match.group("name"), r"-v\d+$", "CACHE_NAME must end in -v<N>")


class ServedAssetTests(unittest.TestCase):
    def test_web_dir_points_to_existing_static_assets(self):
        self.assertTrue((WEB_DIR / "index.html").is_file())

    def test_root_static_assets_are_whitelisted_and_present(self):
        for asset_name in main.ROOT_STATIC_MEDIA_TYPES:
            self.assertTrue((WEB_DIR / asset_name).is_file(), asset_name)


class TestSuiteShapeTests(unittest.TestCase):
    """CI runs ``unittest discover``; pytest-only tests would never execute."""

    def test_no_module_level_pytest_functions(self):
        offenders = [
            path.name
            for path in sorted(TESTS_DIR.glob("test_*.py"))
            if re.search(r"^def test_", path.read_text(encoding="utf-8"), re.MULTILINE)
        ]
        self.assertEqual(
            offenders,
            [],
            "module-level def test_* is invisible to `unittest discover` -- "
            "write these as unittest.TestCase methods",
        )

    def test_no_pytest_import(self):
        offenders = [
            path.name
            for path in sorted(TESTS_DIR.glob("test_*.py"))
            if re.search(r"^import pytest|^from pytest\b", path.read_text(encoding="utf-8"), re.M)
        ]
        self.assertEqual(
            offenders,
            [],
            "tests must run under the stdlib runner; use unittest.skipUnless instead of "
            "pytest.importorskip",
        )


if __name__ == "__main__":
    unittest.main()
