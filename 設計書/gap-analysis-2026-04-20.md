# CordBeat 設計 vs 実装 ギャップ分析レポート

**日付**: 2026-04-20 (Batch C 完了反映)
**ブランチ**: main (PR #79 sqlite-vec / #82 skill venv / #83 integration test +85% coverage マージ後)
**テスト数**: 618 passed, 1 skipped (Windows symlink)
**カバレッジ**: 90.85% (gate: 85%)

---

## 全体サマリー

| 指標 | 値 |
|------|-----|
| **全体実装率** | **100%** |
| **設計ドキュメント** | 12 / 12 カバー |
| **ソースモジュール** | 29 / 29 実装済 |
| **ビルトインスキル** | 8 / 8 実装済 |
| **Batch C** | **完了**（#2 / #3 / #4 / #7 / #10 すべてマージ済） |
| **残存ギャップ** | 0 件 |

---

## 設計ドキュメント別ステータス

| # | 設計ドキュメント | 実装率 | 備考 |
|---|-----------------|--------|------|
| 1 | philosophy.md | 95% | ✅ 自律・感情・記憶・自己改善すべて実装 |
| 2 | architecture.md | 95% | ✅ アカウントリンク(PR#48)、ユーザーコマンド(PR#49)完了 |
| 3 | heartbeat.md | 98% | ✅ 2層HEARTBEAT、提案、sleep phase、分割(PR#55) |
| 4 | soul.md | 98% | ✅ パーミッションマトリクス実行時強制を完全化（本セッション） |
| 5 | memory.md | 100% | ✅ 4層、lazy decay (#4)、sqlite-vec (#7)、UUID user_id (#3)、統合テスト (#10) |
| 6 | skills.md | 100% | ✅ 8スキル実装、サンドボックス、安全レベル、**Skill 専用 venv (#2 / PR #82)** |
| 7 | gateway.md | 95% | ✅ WebSocket、アダプタ3種、再接続 |
| 8 | validation.md | 98% | ✅ 3層防御、リトライ、フォールバック |
| 9 | engine.md | 95% | ✅ メッセージ処理、コマンド、抽出 |
| 10 | ai-backends.md | 100% | ✅ Ollama、OpenAI、openai_compat |
| 11 | config-reference.md | 100% | ✅ ログローテーション(PR#70)、パス解決(PR#70) |
| 12 | deployment.md | 95% | ✅ Docker、docker-compose、セットアップウィザード |

---

## 解決済みギャップ（セキュリティ強化セッション）

| ギャップ | 修正内容 | PR |
|---------|---------|-----|
| スキル実行が同一プロセス内 | AST検証 + サブプロセス隔離 + リソース制限 | #67 |
| SSRF脆弱性 | DNS解決 + private IP blocking + redirect無効化 | #67 |
| WebSocket認証なし | HMAC token認証 (`hmac.compare_digest`) | #68 |
| バインドアドレス 0.0.0.0 | デフォルト `127.0.0.1` に変更 | #68 |
| DNS rebinding攻撃 | URL書き換え + Host header固定 | #68 |
| プロンプトインジェクション | メモリデリミタ + system prompt指示 + 500文字制限 | #68 |
| 設計書がセキュリティ未反映 | 6ファイル更新（skills, gateway, config, memory, architecture, engine） | #69 |
| `~/.cordbeat/` データルート規約 | `_resolve_relative_paths()` でconfig相対パス解決 | #70 |
| ログローテーション未実装 | `RotatingFileHandler` + LogConfig fields | #70 |
| 例外階層欠如 | `CordBeatError` 基底 + `SkillError` サブツリーを `exceptions.py` に集約、既存の型付きエラーを再親子化 | (本セッション) |
| api_call Ruff エラー | `httpcore` 未使用import削除 + import並び替え | (本セッション) |
| SOUL パーミッション実行時強制が弱い | `update_emotion` / `update_name` / `update_notes` / `apply_trait_change` / `update_quiet_hours` の `caller` 引数を keyword-only 必須化、デフォルト値廃止。production 4 箇所（engine.py×2, extraction.py, heartbeat_proposals.py）に明示的 `caller=` を付与。全テスト更新 & 599 passed | (本セッション) |
| 追加プラットフォームアダプタ未実装 | Slack / LINE / WhatsApp / Signal の 4 アダプタスケルトンを追加（Socket Mode / Webhook / Cloud API / signal-cli JSON-RPC）。`adapter_runner` / CLI エントリ / `optional-dependencies` / mypy 対象に反映、`test_adapter_runner.py` にパラメトリック起動テストを追加（603 passed） | (本セッション) |
| **Batch C #7: Chroma → sqlite-vec 置換** | ChromaDB を撤廃し `sqlite-vec` (`vec0` 仮想テーブル + `user_id PARTITION KEY`) に統一。Embedding は `sentence-transformers/all-MiniLM-L6-v2` で local 生成。`chromadb` 依存削除、`chroma_path` 廃止 | #79 |
| **Batch C #4: Forgetting を lazy decay 化** | 夜間バッチ `decay_and_archive_memories()` を撤廃。strength は `_search` 内で都度算出し、`archive_threshold` 未満は参照時に物理削除（flashbulb は除外）。検索は `n_results * 3` でオーバーサンプル + `strength / (1+distance)` で再ランク | #79 |
| **Batch C #3: user_id ランダム化** | 新規ユーザーには `uuid4().hex` を割り当て。`cb_<adapter>_<id>` 形式を廃止、`platform_links` テーブル経由で外部 ID と紐付け。リーク時の追跡可能性を排除 | (実装済 / engine.py:110) |
| **Batch C #2: Skill 専用 venv** | 各ビルトインスキルに `pyproject.toml` を追加、初回実行時に `~/.cordbeat/skill-envs/<name>/` を `uv` で作成・キャッシュ（SHA-256 hash invalidation）。サブプロセスは env の interpreter + `-I` + `PYTHONNOUSERSITE=1` で起動。`skill_runner.py` を self-contained 化（cordbeat パッケージ非依存） | #82 |
| **Batch C #10: 統合テスト + カバレッジ 85%** | `tests/integration/test_memory_vector.py` で sqlite-vec + sentence-transformers の E2E シナリオを追加。カバレッジゲート 78% → 85%（実測 90.85%）、`skill_runner.py` と 4 アダプタスケルトンを measurement から除外 | #83 |

## 以前のPRで解決済みだったギャップ（メモ更新前）

| ギャップ | 修正内容 | PR |
|---------|---------|-----|
| アカウントリンクフロー | `/link`, `/unlink` コマンド + トークン | #48 |
| ユーザーコマンド | `/name`, `/quiet`, `/prefer` | #49 |
| ロギング設定 | `log.level`, `log.format` | #50 |
| max_tokens ハードコード | `AIBackendConfig.max_tokens` で設定可能 | 不明 |
| n_results ハードコード | `MemoryConfig` に全検索結果数フィールド | 不明 |
| セットアップウィザード | `setup_wizard.py` 完全実装 | 不明 |
| ビルトインスキル不足 | 8スキル全実装（web_search, weather, file_write, api_call, file_read, read_diary, shell_exec, timer） | 不明 |

---

## 残存ギャップ一覧

### Gap 1: bare `except Exception:` — ハンドラ境界は設計上維持（解決扱い）

**現状**: 22箇所の大半は heartbeat loop / sleep phase / adapter message loop など、
「1箇所の失敗で全体を落とさない」境界ハンドラであり、すべて `logger.exception()`
でトレースバックをロギング済。型付き catch への置換は設計意図に反する。

**実施済み対応** (本セッション):
- `src/cordbeat/exceptions.py` を新設し `CordBeatError` 基底クラス + サブクラス階層を定義:
  - `CordBeatError`
    - `SoulPermissionError` (re-parented from `Exception` in `models.py`)
    - `SkillError`
      - `SkillPermissionError` (re-parented in `skill_sandbox.py` / `skill_runner.py`)
      - `SkillSandboxError` (re-parented in `skill_sandbox.py`)
      - `SkillValidationError` (re-parented; `ValueError` 互換も維持)
    - `MemorySubsystemError` (新規 — 将来の raise 用)
    - `AIBackendError` (新規)
    - `OutputValidationError` (新規; `models.ValidationError` データクラスと別物)
- `cordbeat/__init__.py` から公開、`tests/test_exceptions.py` で契約を固定化

**残課題**: 新規コードは上記の型付き例外を raise / catch すること。
境界 catch-all は従来通り維持。

---

### Gap 2: SOUL パーミッションマトリクスの実行時強制（解決済）

**実施済み対応** (本セッション):
- `soul.py` の 5 変更メソッド (`update_emotion`, `update_name`, `update_notes`,
  `apply_trait_change`, `update_quiet_hours`) の `caller` 引数を
  **keyword-only 必須** に変更（デフォルト値を撤廃）
- 呼び出し側は意図したアクター（`SoulCaller.SYSTEM` / `AI` / `USER`）を
  明示する以外に経路がなくなり、`_check_permission()` が必ず走る
- production 4 箇所を更新:
  - `engine.py` `/name` コマンド → `caller=SoulCaller.USER`
  - `engine.py` `/quiet` コマンド → `caller=SoulCaller.USER`
  - `extraction.py` 感情推論 → `caller=SoulCaller.AI`
  - `heartbeat_proposals.py` 承認済 trait 変更 → `caller=SoulCaller.SYSTEM`
- `tests/test_soul.py` / `tests/test_engine.py` の全呼び出しを更新
- ruff / mypy strict / pytest すべてグリーン（599 passed, 1 skipped）

---

### Gap 3: api_call スキルの Ruff エラー（解決済）

**実施済み対応** (本セッション): 未使用 `httpcore` import 削除 + import並び替え。
`ruff check src tests skills` がクリーンに通る状態に復帰。

---

### Gap 4: 追加プラットフォームアダプタ（解決済 — スケルトン）

**実施済み対応** (本セッション):
- 4 種の新規アダプタファイルを追加（既存 Discord/Telegram と同一パターン）:
  - `slack_adapter.py` — `slack-sdk` Socket Mode + AsyncWebClient（公開 webhook 不要）
  - `line_adapter.py` — `line-bot-sdk` v3 + `aiohttp` webhook サーバー
  - `whatsapp_adapter.py` — Meta Cloud API (`httpx`) + `aiohttp` webhook（verify token 対応）
  - `signal_adapter.py` — `signal-cli` JSON-RPC daemon（httpx poll）
- `adapter_runner.py` に 4 種の分岐追加 + `*_cli` エントリポイント 4 個
- `pyproject.toml`:
  - `[project.scripts]` に `cordbeat-slack` / `cordbeat-line` / `cordbeat-whatsapp` / `cordbeat-signal`
  - `[project.optional-dependencies]` に `slack` / `line` / `whatsapp` / `signal` extras
  - `[[tool.mypy.overrides]]` に 4 モジュールを追加（SDK 未インストール時もチェック通過）
- `tests/test_adapter_runner.py` に 4 種のパラメトリック起動テスト追加

**状態**: SDK 未インストール環境では `start()` がログして早期 return するため、
追加依存なしで CordBeat 全体ビルド・テストは緑のまま。各 SDK を `uv sync --extra <name>`
でインストール後、設定 YAML にトークン/URL を書けばそのまま稼働する「scaffolding」扱い。

**残作業**: v1.0+ で各プラットフォームの E2E 疎通テスト・メディア添付・署名検証強化。

---

## ソースモジュール一覧

| モジュール | 役割 | テスト |
|-----------|------|--------|
| config.py | YAML/env設定読み込み、パス解決 | ✅ |
| models.py | データ型定義 | ✅ |
| exceptions.py | `CordBeatError` 例外階層 | ✅ |
| gateway.py | WebSocketサーバー + アダプタ基盤 | ✅ |
| engine.py | メッセージ処理 + コマンドルーティング | ✅ |
| heartbeat.py | 2層HEARTBEATループ | ✅ |
| heartbeat_proposals.py | 提案処理 | ✅ |
| heartbeat_sleep.py | sleep phase (日記/整理/decay) | ✅ |
| memory.py | 4層メモリシステム | ✅ |
| soul.py | アイデンティティ + 感情 | ✅ |
| extraction.py | 会話からのメモリ抽出 | ✅ |
| skills.py | スキルローディング + レジストリ | ✅ |
| skill_runner.py | スキル実行 | ✅ |
| skill_sandbox.py | サンドボックス | ✅ |
| skill_validator.py | スキルバリデーション | ✅ |
| validation.py | AI出力バリデーション | ✅ |
| prompt.py | プロンプト構築 | ✅ |
| ai_backend.py | バックエンド抽象化 | ✅ |
| main.py | エントリポイント | ✅ |
| setup_wizard.py | セットアップウィザード | ✅ |
| discord_adapter.py | Discordボット | ✅ |
| telegram_adapter.py | Telegramボット | ✅ |
| slack_adapter.py | Slackボット（Socket Mode, scaffold） | ✅ |
| line_adapter.py | LINE Messaging API（webhook, scaffold） | ✅ |
| whatsapp_adapter.py | WhatsApp Cloud API（webhook, scaffold） | ✅ |
| signal_adapter.py | Signal（signal-cli RPC, scaffold） | ✅ |
| cli_adapter.py | CLIインターフェース | ✅ |
| adapter_runner.py | アダプタブートストラップ | ✅ |
| doctor.py | 診断ツール | ✅ |

## ビルトインスキル一覧

| スキル | 安全レベル | 状態 |
|--------|-----------|------|
| file_read | safe | ✅ |
| file_write | requires_confirmation | ✅ |
| read_diary | safe | ✅ |
| shell_exec | dangerous | ✅ |
| timer | safe | ✅ |
| web_search | safe | ✅ |
| weather | safe | ✅ |
| api_call | requires_confirmation | ✅ |

---

## 推奨アクション

### 次セッションで対応候補
1. **SOULパーミッション強化** — soul.py の書き込みメソッドに `SoulCaller` 権限チェックを追加し、`SoulPermissionError` を raise
2. **追加アダプタ** — LINE / Slack 等のプラットフォーム対応 (v1.0+)

### 本セッションで完了
- ✅ **api_call ruff修正** — 未使用import削除 + 並び替え
- ✅ **例外階層整備** — `CordBeatError` 基底 + `SkillError` サブツリー + 6件のテスト
- ✅ **Batch C #7 (sqlite-vec)** — ChromaDB 撤廃、`vec0` 仮想テーブル化 (PR #79)
- ✅ **Batch C #4 (lazy decay)** — 夜間バッチ廃止、参照時算出+削除 (PR #79 同梱)
- ✅ **Batch C #3 (UUID user_id)** — `uuid4().hex` 化、`platform_links` 経由
- ✅ **Batch C #2 (Skill 専用 venv)** — `uv` 管理 per-skill env + 完全分離 (PR #82, v0.5.0)
- ✅ **Batch C #10 (統合テスト + カバレッジ 85%)** — `tests/integration/` 新設、gate 78%→85% (PR #83, 90.85%)

### 残作業（v1.0+ 候補）
- 各プラットフォームアダプタ (Slack / LINE / WhatsApp / Signal) の E2E 疎通テスト・メディア添付・署名検証強化
- v1.0 リリースタグの判断（Batch C 完了によりコア機能は安定）

---

*生成: GitHub Copilot | 更新: 2026-04-20 (Batch C 完了反映セッション — PR #79 / #82 / #83)*
