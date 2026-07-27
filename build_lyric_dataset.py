"""CLI shim — implementation lives in breeze_elf.dataset_builder.

Run with the project venv: ``uv run python build_lyric_dataset.py --help``.
"""

from breeze_elf.dataset_builder import main

if __name__ == "__main__":
    raise SystemExit(main())
