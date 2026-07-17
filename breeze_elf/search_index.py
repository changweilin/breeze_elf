"""Cross-transcript full-text search over the saved ``remote_transcripts`` bundles.

A single SQLite FTS5 table (``trigram`` tokenizer) built from the ``.txt`` files on
disk, so the index is disposable and rebuilds from files alone. Trigram is the crux:
the default ``unicode61`` tokenizer leaves a whole Chinese sentence as one token, so
substring queries like ``在乎`` return nothing — trigram makes CJK substring search
work with no extra dependency. Bodies and queries are OpenCC-normalized (opencc is
already a base dependency) so 繁/簡 queries cross the script boundary.

Everything degrades gracefully: if the runtime SQLite lacks FTS5/trigram the index
reports :attr:`available` = ``False`` and every operation becomes a no-op, mirroring
how the Silero VAD / enhancement stages self-disable rather than break the app.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from .storage import TranscriptRecord, iter_transcript_ids, load_transcript

logger = logging.getLogger(__name__)

# MATCH needs >= this many chars for the trigram tokenizer; shorter CJK queries
# (一/兩 字) fall back to a LIKE substring scan.
_MIN_MATCH_CHARS = 3
_SNIPPET_RADIUS = 12
# Bump when the table shape changes. The index is a disposable cache rebuilt from
# the .txt/.json files, so a mismatch just drops and re-creates the table (an old
# index from a prior schema would otherwise make every query error out).
_SCHEMA_VERSION = 2

# Column order shared by every write path.
_INSERT_SQL = (
    "INSERT INTO transcripts (doc_id, filename, created_at, has_audio, has_jianpu, "
    "mtime, raw_title, raw_body, title, body) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
# Display columns fetched for every result row (raw, on-disk text).
_SELECT_COLS = "doc_id, filename, created_at, has_audio, has_jianpu, raw_title, raw_body"
# Newest-first by the numeric mtime; created_at strings carry mixed tz offsets and
# don't sort chronologically as text.
_ORDER_RECENT = "ORDER BY CAST(mtime AS REAL) DESC"


def _make_search_converter():
    """OpenCC simplified→traditional converter used to canonicalize both stored
    text and queries so 繁/簡 match each other. ``None`` (identity) when opencc is
    unavailable — search still works, just script-sensitive."""
    try:
        from opencc import OpenCC
    except Exception:
        return None
    for config_name in ("s2t", "s2tw"):
        try:
            return OpenCC(config_name)
        except Exception:
            continue
    return None


class SearchIndex:
    name = "sqlite-fts5-trigram"

    def __init__(self, db_path: str | Path, *, enabled: bool = True) -> None:
        self.db_path = Path(db_path).expanduser()
        self.available = False
        self.unavailable_reason: str | None = None
        # Serialize writers: SQLite is single-writer, and upsert/sync run from
        # executor threads. Reads open their own short-lived connection (WAL).
        self._write_lock = threading.Lock()
        self._converter = _make_search_converter()

        if not enabled:
            self.unavailable_reason = "disabled (BREEZE_SEARCH_ENABLED=0)"
            return
        try:
            self._ensure_schema()
            self.available = True
        except Exception as exc:  # sqlite without FTS5/trigram, or an unwritable dir
            self.unavailable_reason = f"FTS5/trigram unavailable: {exc}"
            logger.warning("transcript search disabled — %s", self.unavailable_reason)

    # -- lifecycle -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")
        # Erase freed content on DELETE so a purged transcript's plaintext doesn't
        # linger in the index file's freelist (the index is a second at-rest copy).
        conn.execute("PRAGMA secure_delete=ON")
        return conn

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            # A pre-existing index from an older schema version won't have the new
            # columns; drop it so it is rebuilt from disk rather than erroring queries.
            if conn.execute("PRAGMA user_version").fetchone()[0] != _SCHEMA_VERSION:
                conn.execute("DROP TABLE IF EXISTS transcripts")
            # raw_title/raw_body are UNINDEXED and hold the on-disk text verbatim —
            # they are what gets displayed, so the result preview always matches the
            # transcript that opens. title/body are the OpenCC-normalized copies that
            # are actually indexed, so 繁/簡 queries cross without altering display.
            # Raises when the runtime SQLite lacks FTS5/trigram — the availability
            # signal the caller degrades on.
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS transcripts USING fts5("
                "doc_id UNINDEXED, filename UNINDEXED, created_at UNINDEXED, "
                "has_audio UNINDEXED, has_jianpu UNINDEXED, mtime UNINDEXED, "
                "raw_title UNINDEXED, raw_body UNINDEXED, "
                "title, body, tokenize='trigram')"
            )
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            conn.commit()

    def _norm(self, text: str) -> str:
        if not text or self._converter is None:
            return text
        try:
            return self._converter.convert(text)
        except Exception:
            return text

    # -- writes --------------------------------------------------------------

    def _row_from_record(self, record: TranscriptRecord) -> tuple:
        return (
            record.id,
            record.filename,
            record.created_at,
            "1" if record.has_audio else "0",
            "1" if record.has_jianpu else "0",
            str(record.mtime),
            record.title,  # raw_title (display)
            record.text,  # raw_body (display)
            self._norm(record.title),  # title (indexed / match)
            self._norm(record.text),  # body (indexed / match)
        )

    def _insert(self, conn: sqlite3.Connection, record: TranscriptRecord) -> None:
        conn.execute("DELETE FROM transcripts WHERE doc_id = ?", (record.id,))
        conn.execute(_INSERT_SQL, self._row_from_record(record))

    def upsert(self, record: TranscriptRecord) -> None:
        """Index (or re-index) one transcript. Best-effort: an index failure must
        never fail the save that triggered it — the caller ignores exceptions and
        a later :meth:`sync` self-heals from the files on disk."""
        if not self.available:
            return
        with self._write_lock, closing(self._connect()) as conn:
            self._insert(conn, record)
            conn.commit()

    def sync(self, storage_dir: str | Path) -> int:
        """Reconcile the index with the transcript files: (re)index any whose
        ``.txt`` mtime changed and drop rows whose file is gone. Cheap for the few
        hundred files this app produces — only changed files are re-read. Returns
        the current indexed count.

        Never raises: a locked DB (a concurrent writer in the multi-server topology)
        or one unreadable file must degrade search, not 500 the request — so the
        whole reconcile is guarded and each file is read independently."""
        if not self.available:
            return 0
        try:
            with self._write_lock, closing(self._connect()) as conn:
                indexed = {
                    row[0]: row[1]
                    for row in conn.execute("SELECT doc_id, mtime FROM transcripts")
                }
                on_disk = set(iter_transcript_ids(storage_dir))

                for gone in indexed.keys() - on_disk:
                    conn.execute("DELETE FROM transcripts WHERE doc_id = ?", (gone,))

                for doc_id in on_disk:
                    try:
                        record = load_transcript(storage_dir, doc_id)
                    except OSError:  # one bad/removed file must not abort the sweep
                        continue
                    if record is None or indexed.get(doc_id) == str(record.mtime):
                        continue  # missing or unchanged
                    self._insert(conn, record)
                conn.commit()
                return int(conn.execute("SELECT count(*) FROM transcripts").fetchone()[0])
        except sqlite3.Error as exc:
            logger.warning("transcript index sync failed: %s", exc)
            return self.indexed_count()

    def remove(self, doc_id: str) -> None:
        if not self.available:
            return
        with self._write_lock, closing(self._connect()) as conn:
            conn.execute("DELETE FROM transcripts WHERE doc_id = ?", (doc_id,))
            conn.commit()

    def indexed_count(self) -> int:
        if not self.available:
            return 0
        try:
            with closing(self._connect()) as conn:
                return int(conn.execute("SELECT count(*) FROM transcripts").fetchone()[0])
        except sqlite3.Error:
            return 0

    # -- reads ---------------------------------------------------------------

    def search(self, query: str, limit: int = 50) -> list[dict]:
        """Query router: >=3 chars → FTS5 MATCH ranked by bm25; 1-2 chars → LIKE
        substring scan; empty → browse newest. Matching runs on the normalized
        (繁/簡-folded) columns and is wrapped as a literal phrase so FTS5 operators
        in user input can't change or break the match; the displayed title/snippet
        come from the raw on-disk columns so the preview matches the opened file."""
        if not self.available:
            return []
        raw_query = (query or "").strip()
        normalized = self._norm(raw_query)
        limit = max(1, min(int(limit), 200))
        try:
            with closing(self._connect()) as conn:
                if not normalized:
                    return self._browse(conn, limit)
                if len(normalized) >= _MIN_MATCH_CHARS:
                    return self._match(conn, normalized, raw_query, limit)
                return self._like(conn, normalized, raw_query, limit)
        except sqlite3.Error as exc:
            logger.warning("transcript search query failed: %s", exc)
            return []

    def _browse(self, conn: sqlite3.Connection, limit: int) -> list[dict]:
        rows = conn.execute(
            f"SELECT {_SELECT_COLS} FROM transcripts {_ORDER_RECENT} LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_result(row, row[6][:80]) for row in rows]

    def _match(self, conn, normalized: str, raw_query: str, limit: int) -> list[dict]:
        phrase = '"' + normalized.replace('"', '""') + '"'
        rows = conn.execute(
            f"SELECT {_SELECT_COLS} FROM transcripts "
            "WHERE transcripts MATCH ? ORDER BY bm25(transcripts) LIMIT ?",
            (phrase, limit),
        ).fetchall()
        return [self._row_to_result(row, _manual_snippet(row[6], raw_query)) for row in rows]

    def _like(self, conn, normalized: str, raw_query: str, limit: int) -> list[dict]:
        pattern = f"%{_escape_like(normalized)}%"
        rows = conn.execute(
            f"SELECT {_SELECT_COLS} FROM transcripts "
            "WHERE body LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\' "
            f"{_ORDER_RECENT} LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return [self._row_to_result(row, _manual_snippet(row[6], raw_query)) for row in rows]

    @staticmethod
    def _row_to_result(row: tuple, snippet: str | None) -> dict:
        return {
            "id": row[0],
            "filename": row[1],
            "createdAt": row[2],
            "hasAudio": row[3] == "1",
            "hasJianpu": row[4] == "1",
            "title": row[5],  # raw_title
            "snippet": snippet,
        }


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _manual_snippet(body: str, needle: str) -> str:
    """A '…in-context…' excerpt around the first match, for the LIKE path where
    FTS5's snippet() isn't available."""
    index = body.find(needle)
    if index < 0:
        return body[: _SNIPPET_RADIUS * 4]
    stop = index + len(needle)
    start = max(0, index - _SNIPPET_RADIUS)
    end = min(len(body), stop + _SNIPPET_RADIUS)
    excerpt = f"{body[start:index]}[{body[index:stop]}]{body[stop:end]}"
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(body) else ""
    return f"{prefix}{excerpt}{suffix}"


def build_search_index(db_path: str | Path, *, enabled: bool = True) -> SearchIndex:
    return SearchIndex(db_path, enabled=enabled)
