import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from breeze_elf import main
from breeze_elf.search_index import build_search_index
from breeze_elf.storage import load_transcript, save_transcript

try:
    from starlette.testclient import TestClient
except Exception:  # pragma: no cover - starlette always present with fastapi
    TestClient = None


def _index_for(directory):
    idx = build_search_index(os.path.join(directory, "_index.sqlite3"))
    idx.sync(directory)
    return idx


class SearchIndexTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        save_transcript(
            "今天開會討論在乎的專案進度", self.dir, title="會議一",
            structured={"title": "會議一", "blocks": []},
        )
        save_transcript("明天的行程與英文 meeting notes", self.dir, title="行程")
        self.idx = _index_for(self.dir)

    def _titles(self, q):
        return {r["id"] for r in self.idx.search(q, 10)}

    def test_available(self):
        self.assertTrue(self.idx.available, self.idx.unavailable_reason)

    def test_two_char_cjk_matches_via_like(self):
        # The whole reason for trigram: a 2-char substring 'in the middle' must hit.
        self.assertEqual(len(self.idx.search("在乎", 10)), 1)

    def test_match_returns_snippet_with_highlight(self):
        results = self.idx.search("開會討論", 10)  # >=3 chars -> FTS5 MATCH
        self.assertEqual(len(results), 1)
        self.assertIn("[", results[0]["snippet"])  # snippet() brackets the hit

    def test_latin_term_matches(self):
        self.assertEqual(len(self.idx.search("meeting", 10)), 1)

    def test_title_is_searchable(self):
        # Title only persists when the structured .json is written (doc 1).
        self.assertEqual(len(self.idx.search("會議", 10)), 1)

    def test_traditional_simplified_cross(self):
        if self.idx._converter is None:
            self.skipTest("opencc not available")
        # A Simplified query must find a Traditional document (讨论 -> 討論).
        self.assertEqual(len(self.idx.search("讨论", 10)), 1)

    def test_empty_query_browses_newest_first(self):
        results = self.idx.search("", 10)
        self.assertEqual(len(results), 2)
        # Newest-first by real instant (ordering is by numeric mtime, not by the
        # createdAt strings which carry mixed tz offsets and don't sort as text).
        first = datetime.fromisoformat(results[0]["createdAt"])
        second = datetime.fromisoformat(results[1]["createdAt"])
        self.assertGreaterEqual(first, second)

    def test_no_match_is_empty(self):
        self.assertEqual(self.idx.search("完全找不到xyzzy", 10), [])

    def test_fts5_operators_in_query_are_literal(self):
        # 開會討 lives in doc 1, 明天的 in doc 2. As a literal phrase this matches
        # NEITHER (0); if the OR operator leaked through unescaped it would match
        # BOTH (2). Asserting 0 actually locks the phrase-escaping.
        self.assertEqual(len(self.idx.search("開會討 OR 明天的", 10)), 0)

    def test_display_text_is_raw_not_normalized(self):
        # A Simplified transcript must be findable (via 繁/簡 folding) yet display
        # its ORIGINAL characters — the preview must match the on-disk file.
        if self.idx._converter is None:
            self.skipTest("opencc not available")
        save_transcript(
            "我们讨论了方案", self.dir, title="会议", structured={"title": "会议", "blocks": []}
        )
        self.idx.sync(self.dir)
        hits = self.idx.search("讨论", 10)
        self.assertTrue(any(h["title"] == "会议" for h in hits))  # raw Simplified, not 會議
        hit = next(h for h in hits if h["title"] == "会议")
        self.assertIn("讨论", hit["snippet"])  # snippet from raw body

    def test_sync_degrades_on_unreadable_file(self):
        # A file that fails to read mid-sweep must be skipped, not abort the reconcile.
        import breeze_elf.search_index as si

        real = si.load_transcript
        calls = {"n": 0}

        def flaky(storage_dir, doc_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("boom")
            return real(storage_dir, doc_id)

        with patch.object(si, "load_transcript", flaky):
            # rebuild from scratch so both files are re-read this sweep
            fresh = build_search_index(os.path.join(self.dir, "_index2.sqlite3"))
            count = fresh.sync(self.dir)  # must not raise
        self.assertGreaterEqual(count, 1)

    def test_sync_drops_deleted_file(self):
        # delete the 'meeting' transcript and re-sync
        for path in Path(self.dir).glob("*.txt"):
            if "meeting" in path.read_text(encoding="utf-8"):
                path.unlink()
        self.idx.sync(self.dir)
        self.assertEqual(len(self.idx.search("meeting", 10)), 0)
        self.assertEqual(self.idx.indexed_count(), 1)

    def test_upsert_indexes_new_transcript(self):
        stored = save_transcript("嶄新的一段錄音內容", self.dir)
        record = load_transcript(self.dir, stored.id)
        self.idx.upsert(record)
        self.assertEqual(len(self.idx.search("嶄新", 10)), 1)

    def test_stale_schema_is_rebuilt_not_errored(self):
        # A pre-existing index from the OLD (pre-raw-columns) schema must be dropped
        # and rebuilt, or every query would error on the missing columns.
        import sqlite3

        stale = os.path.join(self.dir, "_stale.sqlite3")
        con = sqlite3.connect(stale)
        con.execute(
            "CREATE VIRTUAL TABLE transcripts USING fts5("
            "doc_id UNINDEXED, filename UNINDEXED, created_at UNINDEXED, "
            "has_audio UNINDEXED, has_jianpu UNINDEXED, mtime UNINDEXED, "
            "title, body, tokenize='trigram')"  # old 8-column shape, user_version 0
        )
        con.execute(
            "INSERT INTO transcripts (doc_id, filename, created_at, has_audio, "
            "has_jianpu, mtime, title, body) VALUES "
            "('gone', 'gone.txt', '', '0', '0', '0', '舊標題', '舊資料')"
        )
        con.commit()
        con.close()

        idx = build_search_index(stale)  # must migrate, not carry the old table
        self.assertTrue(idx.available)
        idx.sync(self.dir)
        self.assertEqual(idx.search("舊資料", 10), [])  # old row dropped
        self.assertEqual(len(idx.search("開會討論", 10)), 1)  # real files re-indexed


class DisabledIndexTests(unittest.TestCase):
    def test_disabled_is_noop(self):
        d = tempfile.mkdtemp()
        idx = build_search_index(os.path.join(d, "x.sqlite3"), enabled=False)
        self.assertFalse(idx.available)
        self.assertEqual(idx.search("在乎", 5), [])
        self.assertEqual(idx.indexed_count(), 0)
        idx.sync(d)  # must not raise


@unittest.skipIf(TestClient is None, "starlette TestClient unavailable")
class SearchEndpointTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.stored = save_transcript(
            "端到端搜尋測試內容", self.dir, title="端到端",
            structured={"title": "端到端", "blocks": [{"text": "端到端搜尋測試內容"}]},
        )
        self.idx = _index_for(self.dir)
        self._patches = [
            patch.object(main, "search_index", self.idx),
            patch.object(main, "_remote_storage_dir", lambda: Path(self.dir)),
        ]
        for p in self._patches:
            p.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_search_route_is_not_shadowed_by_id(self):
        # /api/transcripts/search must route to search, not read {id="search"}.
        resp = self.client.get("/api/transcripts/search", params={"q": "搜尋測試"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["results"][0]["id"], self.stored.id)

    def test_read_rejects_path_traversal(self):
        resp = self.client.get("/api/transcripts/..%2f..%2fetc%2fpasswd")
        self.assertIn(resp.status_code, (400, 404))  # never a 200 file read

    def test_read_missing_returns_404(self):
        resp = self.client.get("/api/transcripts/does-not-exist")
        self.assertEqual(resp.status_code, 404)

    def test_read_roundtrips_transcript(self):
        resp = self.client.get(f"/api/transcripts/{self.stored.id}")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["text"], "端到端搜尋測試內容")
        self.assertEqual(body["title"], "端到端")
        self.assertTrue(isinstance(body["blocks"], list))

    def test_list_browses(self):
        resp = self.client.get("/api/transcripts")
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.json()["count"], 1)


@unittest.skipIf(TestClient is None, "starlette TestClient unavailable")
class SearchDisabledEndpointTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.stored = save_transcript(
            "停用搜尋時仍可直接讀取", self.dir, title="讀取",
            structured={"title": "讀取", "blocks": [{"text": "停用搜尋時仍可直接讀取"}]},
        )
        disabled = build_search_index(os.path.join(self.dir, "_x.sqlite3"), enabled=False)
        self._patches = [
            patch.object(main, "search_index", disabled),
            patch.object(main, "_remote_storage_dir", lambda: Path(self.dir)),
        ]
        for p in self._patches:
            p.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_search_and_list_return_503_when_disabled(self):
        search = self.client.get("/api/transcripts/search", params={"q": "x"})
        self.assertEqual(search.status_code, 503)
        self.assertEqual(self.client.get("/api/transcripts").status_code, 503)

    def test_read_still_works_when_search_disabled(self):
        # The read path hits disk directly and must be unaffected by the search gate.
        resp = self.client.get(f"/api/transcripts/{self.stored.id}")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
