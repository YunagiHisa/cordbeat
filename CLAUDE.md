# CLAUDE.md

## プロジェクト概要

CordBeat — ローカルファースト自律型AIエージェント。感情・記憶・自己改善を持ち、Discord / Telegram / CLI でユーザーと対話する。

### 技術スタック

- **Runtime**: Python 3.11+
- **パッケージ管理**: `uv`
- **AI バックエンド**: Ollama / OpenAI / openai_compat（llama.cpp 等）
- **永続化**: ChromaDB（ベクトル）+ aiosqlite（構造化）
- **プラットフォーム**: Discord / Telegram / CLI（WebSocket アダプタ経由）
- **テスト**: pytest + pytest-asyncio（599 tests, 1 skip）
- **Lint / 型**: ruff + mypy (strict)
- **ビルド**: `uv run pytest` / `uv run ruff check` / `uv run mypy src`

### ファイル構造

```
src/cordbeat/
├── main.py                # エントリポイント
├── config.py              # YAML/env設定、パス解決
├── models.py              # データ型定義
├── exceptions.py          # CordBeatError 例外階層
├── gateway.py             # WebSocketサーバー + アダプタ基盤
├── engine.py              # メッセージ処理 + コマンドルーティング
├── heartbeat.py           # 2層HEARTBEATループ
├── heartbeat_proposals.py # 提案処理
├── heartbeat_sleep.py     # sleep phase（日記/整理/decay）
├── memory.py              # 4層メモリシステム
├── soul.py                # アイデンティティ + 感情
├── extraction.py          # 会話からのメモリ抽出
├── skills.py              # スキルローディング + レジストリ
├── skill_runner.py        # サブプロセス側スキル実行
├── skill_sandbox.py       # 親側サンドボックス
├── skill_validator.py     # AST ベース静的検証
├── validation.py          # AI出力バリデーション
├── prompt.py              # プロンプト構築
├── ai_backend.py          # バックエンド抽象化
├── setup_wizard.py        # cordbeat-init ウィザード
├── discord_adapter.py
├── telegram_adapter.py
├── cli_adapter.py
├── adapter_runner.py
└── doctor.py              # 診断ツール
skills/                    # ビルトインスキル（8種）
├── file_read / file_write / read_diary
├── shell_exec / timer
├── web_search / weather / api_call
tests/                     # pytest スイート
設計書/                    # 設計ドキュメント（12種）+ gap-analysis
docs/                      # 公開ドキュメント
```

### キーパターン

- **設計 vs 実装**: `設計書/gap-analysis-*.md` で差分追跡（最新は 2026-04-20、実装率 ~98%）
- **パス解決**: config 相対パスは `_resolve_relative_paths()` で `~/.cordbeat/` ルートに解決
- **スキル安全レベル**: `safe` / `requires_confirmation` / `dangerous` — `dangerous` はサブプロセス + AST 検証 + リソース制限
- **SSRF防御**: api_call は DNS 解決 + private IP ブロック + redirect 無効化 + Host header 固定
- **WebSocket 認証**: HMAC token（`hmac.compare_digest`）、デフォルト bind は `127.0.0.1`
- **例外階層**: `CordBeatError` を基底に `SkillError` / `SoulPermissionError` 等。境界 catch-all（heartbeat loop / adapter loop）は設計上維持、内部は型付き raise
- **メモリ**: 4層（Profile / Semantic / Episodic / Conversation）+ チェーンリコール + decay
- **HEARTBEAT**: 2層ループ（thinking / sleep phase）、提案 atomic commit

## フォローアップの処理方針

- タスク完了時、必ず `AskUserQuestion`（この環境では `ask_user` ツール）を使って次のアクション候補を選択肢として提示すること
- テキストで「何かありますか？」と聞くのではなく、必ず選択肢UIで提示すること
- 選択肢には必ず「特になし（完了）」を含めること
- ユーザーが選択するまで次のアクションに進まないこと

## 確認が必要な場面

- ファイルの変更・削除を行う前
- 新しい依存関係を追加する前（`pyproject.toml`）
- テストの実行方針を決める前
- 複数の実装アプローチがある場合
- 曖昧な指示を受けた場合（推測で進めず質問すること）

## AskUserQuestion の使い方ルール

- 各選択肢には簡潔な説明をつけること
- おすすめのオプションがある場合はリストの最初に置き「(Recommended)」をラベルに含めること
- 選択肢は2〜5個程度に収めること
- その他の選択肢で記入できる選択肢も提供すること（freeform 許可）
- この環境での実ツール名は `ask_user` — `AskUserQuestion` 指示の実装先として必ずそれを使うこと
- 最終応答の直前に「ask_user をもう呼んだか」「特になし（完了）を入れたか」を必ず自己確認すること

## セッション継続

作業を再開するときは、まず以下を読むこと:

- `設計書/gap-analysis-*.md` - 最新のギャップ分析（設計 vs 実装の差分）
- `CHANGELOG.md` - リリース履歴
- `README.md` - セットアップ手順と全体像

変更があった場合、関連する設計書・gap-analysis を更新すること。

---

## ワークフロー管理

### 1. 計画モード（デフォルト）
- 些細でないタスク（3ステップ以上、またはアーキテクチャ判断が必要）には必ず計画モードに入ること
- 何か問題が発生したら、すぐに停止して再計画すること — 無理に進めない
- 構築だけでなく、検証ステップにも計画モードを使用すること
- 曖昧さを減らすため、最初に詳細な仕様を書くこと

### 2. サブエージェント戦略
- メインのコンテキストウィンドウをクリーンに保つため、サブエージェントを積極的に使用すること
- 調査、探索、並列分析はサブエージェントにオフロードすること
- 複雑な問題には、サブエージェントを通じてより多くの計算リソースを投入すること
- 1つのサブエージェントにつき1つのタスクで集中実行すること

### 3. 自己改善ループ
- ユーザーからの訂正があった場合は必ずパターンを記録すること
- 同じミスを防ぐためのルールを自分のために書くこと
- ミス率が下がるまで、これらの教訓を徹底的に反復改善すること
- セッション開始時に、関連プロジェクトの教訓を確認すること

### 4. 完了前の検証
- 動作を証明せずにタスクを完了とマークしないこと
- `uv run ruff check src tests skills` / `uv run mypy src` / `uv run pytest -q` を必ず緑で通してから完了扱いにすること
- 自問すること：「シニアエンジニアならこれを承認するだろうか？」
- テストを実行し、ログを確認し、正確性を実証すること

### 5. エレガンスの追求（バランス重視）
- 些細でない変更の場合：一時停止して「もっとエレガントな方法はないか？」と問うこと
- 修正が場当たり的に感じる場合：「今知っていることすべてを踏まえて、エレガントな解決策を実装する」
- シンプルで明白な修正にはこれをスキップすること — 過度な設計は避けること
- 提示する前に自分の作業に疑問を投げかけること

### 6. 自律的なバグ修正
- バグレポートを受けたら：ただ修正すること。手取り足取り聞かない
- ログ、エラー、失敗しているテストを指摘して — それから解決すること
- ユーザーからのコンテキスト切り替えはゼロにすること
- 失敗しているCIテストは、やり方を聞かずに修正すること

## タスク管理

1. **まず計画**: 些細でないタスクは計画を立ててから着手すること
2. **計画の確認**: 実装開始前に `exit_plan_mode` または `ask_user` で確認を取ること
3. **進捗追跡**: SQL の `todos` テーブルで `in_progress` / `done` を遷移させること
4. **変更の説明**: 各ステップで高レベルの要約を行うこと
5. **結果の文書化**: 大きな変更は `設計書/gap-analysis-*.md` に反映すること

## 基本原則

- **シンプルさ優先**: すべての変更を可能な限りシンプルにすること。影響を最小限にすること
- **手抜きなし**: 根本原因を見つけること。一時的な修正はしない。シニア開発者の基準で
- **影響の最小化**: 変更は必要な部分のみに触れること。バグの導入を避けること
- **セキュリティ優先**: SSRF / プロンプトインジェクション / DNS rebinding / 権限昇格の観点を常に意識すること（PR #67 / #68 で強化済）

---

## CordBeat ベストプラクティス

### コード品質ゲート（完了前に必ず全部緑）

```powershell
uv run ruff check src tests skills   # lint
uv run mypy src                       # strict 型チェック
uv run pytest -q                      # 全テスト（599+, 1 skip は Windows symlink）
```

- `mypy strict` で失敗する場合、`type: ignore` は**原則禁止**。プラットフォーム固有 API（`resource.setrlimit` 等）のみ `# type: ignore[attr-defined]` を許容
- 新機能には必ずテストを追加（`tests/test_*.py`）
- カバレッジ下限は 70%（`pyproject.toml` の `fail_under`）

### 例外処理

- **raise する側**: `src/cordbeat/exceptions.py` の型付き例外を使うこと
  - `CordBeatError` / `SkillError` / `SoulPermissionError` / `MemorySubsystemError` / `AIBackendError` / `OutputValidationError`
- **catch する側**:
  - 内部ロジックでは具体型で catch（`except SkillPermissionError:`）
  - **境界ハンドラのみ** `except Exception:` + `logger.exception()` を許容
    - heartbeat loop / sleep phase（1箇所の失敗で loop を落とさないため）
    - adapter message loop（1メッセージの失敗で切断しないため）
  - `except Exception: pass` は**禁止**。必ずログを出すこと

### 非同期コード

- `asyncio` ベース。ブロッキング I/O（`requests`, `sqlite3` 直使用）は禁止
- DB は `aiosqlite` 経由
- HTTP は `httpx.AsyncClient` 経由（スキル内含む）
- サブプロセスは `asyncio.create_subprocess_exec`（`skill_sandbox.py` 参照）

### 日時処理

- **タイムゾーン aware な datetime のみ使用**（PR #53 準拠）
- `datetime.now(timezone.utc)` / `datetime.now(ZoneInfo(...))` を使い、`datetime.utcnow()` は**禁止**
- DB シリアライズは ISO 8601（`datetime.isoformat()`）

### パス解決

- config 内の相対パスは `config.py::_resolve_relative_paths()` が `~/.cordbeat/` ルートに解決する
- 新しい設定パスを追加する場合、`_resolve_relative_paths()` の対象リストに追加すること
- `Path` 型を使い、文字列結合は禁止
- Windows 対応：`\\` / `/` 両対応のため `Path` に統一

### ロギング

- モジュール冒頭で `logger = logging.getLogger(__name__)`
- 境界ハンドラ: `logger.exception("context: %s", value)` でトレースバック込みで記録
- 通常: `logger.info` / `logger.debug` を使い分け
- **`print` は禁止**（CLI アダプタのユーザー出力を除く）
- `log.level` / `log.format` / ローテーションは `LogConfig` で設定可能（PR #70）

### セキュリティ

- **外部 HTTP**: 必ず `api_call` スキル or `_resolve_and_check()` 相当のガードを通す
  - 許可スキーム: `http`, `https` のみ
  - private / loopback / link-local / multicast / metadata IP を DNS 解決時点でブロック
  - redirect 無効化（`follow_redirects=False`）
  - DNS rebinding 防止: 解決済み IP に pin + Host header 固定
- **WebSocket**: HMAC token 認証必須、bind は `127.0.0.1` デフォルト
- **スキル実行**:
  - `safe`: プロセス内 OK
  - `requires_confirmation`: 提案キュー経由でユーザー承認
  - `dangerous`: サブプロセス隔離 + AST 検証 + rlimit + timeout
- **プロンプトインジェクション**: メモリ / ユーザー入力は `sanitize(content, max_len=500)` 通過後に prompt に埋め込む。system prompt 側で明示的な指示分離を行う
- **秘密情報**: config ファイル（`config.yaml`）とコードを分離。ログに API key を出さない

### 設計ドキュメント同期

- 仕様変更は `設計書/*.md` を先に更新 → コード → `gap-analysis-YYYY-MM-DD.md` 反映
- 設計書とコードが乖離した場合、gap-analysis に記録して段階的に修正
- 最新の gap-analysis: `設計書/gap-analysis-2026-04-20.md`

### スキル作成

- `skills/<name>/` に `main.py` + `skill.yaml`
- `execute(**kwargs) -> dict[str, Any]` を async で定義（sync も可）
- `skill.yaml` に `safety_level` / `description` / `params` を記載
- `ALLOWED_IMPORTS`（`skill_validator.py`）外のモジュールは使用禁止
- 外部通信は `httpx` のみ（`socket` / `ssl` 直使用禁止）

### コミットメッセージ

- Conventional commits: `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `style:` / `chore:`
- 末尾に必ず:
  ```
  Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
  ```
- PR 番号は merge 時に自動付与（`(#NN)`）

### テスト方針

- `pytest-asyncio` の `asyncio_mode = "auto"` — `@pytest.mark.asyncio` 不要
- fixture は `tests/conftest.py` に集約
- モックは `unittest.mock.AsyncMock` / `MagicMock` を使用
- Windows 固有の skip は `@pytest.mark.skipif(sys.platform == "win32", reason="...")` で明示
- 統合テストは本物の ChromaDB / aiosqlite を使用（temp dir）

### パフォーマンス

- HEARTBEAT interval は `min/max_interval_minutes` で clamp
- メモリ検索の `n_results` は `MemoryConfig` で調整可能
- `max_tokens` / `temperature` は `AIBackendConfig` で設定可能
- LLM 呼び出しには timeout を必ず設定

---

## 最終確認チェック

- 作業完了時は、テキストだけで次アクションを聞かず、必ず `ask_user` を呼ぶこと
- `特になし（完了）` を選択肢に含めること
- ユーザーが選ぶまで次の作業に進めないこと

---

## 直近のセッション記録

### 2026-04-20: 例外階層整備セッション

**完了したギャップ解消**:

1. **Gap 3 (api_call Ruff)**: `skills/api_call/main.py` から未使用 `httpcore` import を削除 + import 並び替え。`ruff check src tests skills` がクリーンに。
2. **Gap 1 (例外階層)**: `src/cordbeat/exceptions.py` を新設し、型付き例外階層を整備:
   - `CordBeatError` (基底)
     - `SoulPermissionError` (models.py から再親子化)
     - `SkillError`
       - `SkillPermissionError` (skill_sandbox.py / skill_runner.py)
       - `SkillSandboxError` (skill_sandbox.py)
       - `SkillValidationError` (skill_validator.py; `ValueError` 互換維持)
     - `MemorySubsystemError` (新規、将来の raise 用)
     - `AIBackendError` (新規)
     - `OutputValidationError` (新規; `models.ValidationError` データクラスとは別物)
   - `cordbeat` パッケージ直下から公開
   - `tests/test_exceptions.py` (6 tests) で契約を固定化
   - 境界の `except Exception:` (heartbeat loop / sleep phase / adapter) は「1箇所の失敗で loop を落とさない」設計意図のため**維持**。既に `logger.exception` 済。

**ドキュメント更新**:
- `設計書/gap-analysis-2026-04-20.md`: 実装率 97% → 98%、残存ギャップ 4 → 2（Medium 1 / Low 1）、解決済みセクションに2件追記、モジュール一覧に `exceptions.py` 追加

**検証**: ruff / mypy strict / pytest すべて緑（599 passed, 1 skipped）。

**コミット**: 未実行（ユーザー指示により保留中）。

**残存ギャップ（次セッション候補）**:
- Gap 2: SOUL パーミッションマトリクスの実行時強制 — `soul.py` 書き込みメソッドに `SoulCaller` 権限チェック追加 → `SoulPermissionError` raise
- Gap 4: 追加プラットフォームアダプタ（LINE / Slack 等、v1.0+）

### 2026-04-20 (続): SOUL パーミッション実行時強制セッション

**完了したギャップ解消**:

- **Gap 2 (SOUL パーミッション実行時強制)**: `soul.py` の 5 変更メソッド
  (`update_emotion` / `update_name` / `update_notes` / `apply_trait_change` /
  `update_quiet_hours`) の `caller` 引数を **keyword-only 必須** に変更
  （デフォルト値を撤廃）。これにより呼び出し側は意図したアクター
  (`SoulCaller.SYSTEM` / `AI` / `USER`) を必ず明示することになり、
  `_check_permission()` が常に実行される。
  - production 4 箇所を更新:
    - `engine.py` `/name` コマンド → `caller=SoulCaller.USER`
    - `engine.py` `/quiet` コマンド → `caller=SoulCaller.USER`
    - `extraction.py` 感情推論 → `caller=SoulCaller.AI`
    - `heartbeat_proposals.py` 承認済 trait 変更 → `caller=SoulCaller.SYSTEM`
  - 全テスト (`test_soul.py`, `test_engine.py`) 更新

**ドキュメント更新**:

- `設計書/gap-analysis-2026-04-20.md`: 実装率 98% → 99%、
  残存ギャップ 2 → 1（Low 1 のみ）、Gap 2 を解決済みセクションへ移動、
  soul.md 実装率 92% → 98%

**検証**: ruff / mypy strict / pytest すべて緑（599 passed, 1 skipped）。

**コミット**: 未実行（ユーザー指示により保留中）。

**残存ギャップ**: Gap 4（追加プラットフォームアダプタ: LINE / Slack 等、v1.0+）のみ。

### 2026-04-20 (続々): 追加プラットフォームアダプタ・スケルトン追加

**完了したギャップ解消**:

- **Gap 4 (LINE / Slack / WhatsApp / Signal アダプタ)**: 4 種のスケルトンを
  既存 Discord/Telegram と同パターンで追加。
  - `src/cordbeat/slack_adapter.py` — `slack-sdk` Socket Mode（webhook 不要）
  - `src/cordbeat/line_adapter.py` — `line-bot-sdk` v3 + aiohttp webhook
  - `src/cordbeat/whatsapp_adapter.py` — Meta Cloud API (httpx) + aiohttp webhook
  - `src/cordbeat/signal_adapter.py` — signal-cli JSON-RPC daemon
  - `adapter_runner.py` に 4 分岐追加 + CLI エントリ 4 個
    (`cordbeat-slack` / `cordbeat-line` / `cordbeat-whatsapp` / `cordbeat-signal`)
  - `pyproject.toml` の `optional-dependencies` に `slack` / `line` / `whatsapp`
    / `signal` extras 追加、`mypy.overrides` にも追加
  - `tests/test_adapter_runner.py` にパラメトリック起動テスト追加

**ポイント**: SDK 未インストール時は `start()` が早期 return するため、追加
依存なしでプロジェクト全体ビルド・テストが緑のまま維持される。各プラット
フォームは `uv sync --extra <name>` + config にトークン/URL 設定ですぐ稼働。

**ドキュメント更新**:

- `設計書/gap-analysis-2026-04-20.md`: 実装率 99% → 99.5%、残ギャップ 1 → 0
  （全 4 ギャップ解消完了）、モジュール一覧に 4 アダプタ追加

**検証**: ruff / mypy strict (30 files) / pytest すべて緑
（**603 passed**, 1 skipped, +4 新規テスト）。

**コミット**: 未実行（ユーザー指示により保留中）。

**残存ギャップ**: なし。全設計ギャップ解消。v1.0+ での残作業は各プラット
フォームの E2E 疎通テスト・メディア添付・署名検証強化。
