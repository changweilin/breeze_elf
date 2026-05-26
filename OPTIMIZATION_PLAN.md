# Breeze Elf 優化計畫進度表

更新日期：2026-05-26

| 編號 | 優先級 | 項目 | 目標 | 狀態 | 驗證方式 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | 高 | 基準量測 | 建立可重複量測的音訊切窗與 ASR 耗時報告 | 完成 | `npm.cmd run bench:mock` |
| P0-2 | 高 | 背壓觀測 | 回報 queue depth、掉窗數與 ASR 等候時間 | 完成 | 單元測試通過，手機端待實測 |
| P0-3 | 高 | 前端背壓保護 | WebSocket 積壓時丟棄舊音訊 chunk，避免延遲無限累積 | 完成 | 單元測試通過，手機端待實測 |
| P0-4 | 高 | 設定可靠性 | `get_settings()` 每次呼叫都讀取目前環境變數 | 完成 | config 單元測試 |
| P1-1 | 中 | 音訊 ring buffer | 降低 Python 與 AudioWorklet 的重複配置與記憶體拷貝 | 完成 | `npm.cmd run bench -- --seconds 120` |
| P1-2 | 中 | ASR worker queue | 讓多 client 共用穩定的 ASR 佇列與取消策略 | 完成 | 單元測試通過，實際並發串流待實測 |
| P1-3 | 中 | VAD 與字幕去重 | 以語音段落為單位送 ASR，減少重複字幕 | 完成 | `npm.cmd run bench -- --segmenter window/vad --seconds 8` |
| P2-1 | 中 | 前端體驗 | 增加複製、下載、延遲、掉窗、麥克風音量提示 | 完成 | 本機 mock 服務通過，手機端待實測 |
| P2-2 | 低 | 打包與 CI | 補 package data、ruff、CI 測試流程 | 完成 | `uv run --extra dev ruff check .`、`uv build --wheel` |

## 本輪執行結果

已完成 P0-1 到 P2-2 的本機驗證。`npm.cmd run test` 通過 22 個單元測試，`npm.cmd run bench:mock` 產生 mock ASR 基準結果，並確認本機 mock 服務 `http://127.0.0.1:8790/health` 回報正常。

P2-2 已補上 wheel 靜態資源打包、ruff 設定與 GitHub Actions CI。手機端與實際並發串流仍需實測。
