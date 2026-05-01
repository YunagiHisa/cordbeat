# Batch C 実施判断 質問票

> **🎉 Batch C 完了** (2026-04-20):
> - **#2 (Skill 専用 venv)** → PR #82 (v0.5.0) ✅
> - **#3 (user_id ランダム化)** → 実装済 (`engine.py:110` で `uuid.uuid4().hex`) ✅
> - **#4 (lazy decay)** → PR #79 (`_memory_vector.py::_search` 内で算出+閾値削除) ✅
> - **#7 (Chroma → sqlite-vec)** → PR #79 ✅
> - **#10 (統合テスト + カバレッジ 85%)** → PR #83 (実測 90.85%) ✅
>
> 以下は意思決定の経緯記録として保管。

Batch A / B は PR #74–#78 でマージ済み。Batch C は影響範囲が大きいため、
着手前に方針を確定させたい項目を以下にまとめる。

各セクションの末尾 **「回答」** 欄に記入してください（自由記述 or 選択）。
全項目を一度に埋める必要はなく、優先度の高いものから回答で構いません。

---

## 項目 #2 — Skill 専用 venv

### 背景
現状、`skills/*/main.py` はホスト Python プロセス（=cordbeat 本体）と同じ
`sys.path` / 依存関係で実行される。Skill が任意の `pip` 依存を宣言しても、
本体の pinned 版と衝突する / サプライチェーン攻撃で本体まで侵される可能性がある。

### 想定する変更
- `skills/<name>/pyproject.toml` を必須化（統一形式、`uv` 管理前提）
- Skill 初回ロード時に `uv venv .venv-skills/<name>/` + `uv pip install -r pyproject.toml` を自動実行
- Skill 実行を subprocess 化（`.venv-skills/<name>/Scripts/python.exe main.py` を spawn）
- IPC は現行の stdin/stdout JSON を踏襲
- 本体の `sys.path` と完全に切り離す（環境変数 `PYTHONNOUSERSITE=1` + venv 内 python のみ使用）

### 質問

**Q2-1. 実行モデル**
- [X] A. 完全に subprocess 化（venv 分離、プロセス分離）
- [ ] B. venv だけ分離し import は動的に切り替え（プロセス共有）
- [ ] C. 本機能は見送り（Skill は信頼できる前提のまま）

**Q2-2. venv 管理**
- [X] A. 初回ロード時に自動作成・キャッシュ
- [ ] B. CLI コマンド (`cordbeat skills install <name>`) で手動作成
- [ ] C. ドキュメントに手順を書くだけ（自動化しない）

**Q2-3. Skill 作成者向けマニフェスト形式**
- [ ] A. `skill.yaml` に `dependencies:` セクションを追加
- [ ] B. 各 Skill ディレクトリに `requirements.txt`
- [X] C. 各 Skill ディレクトリに最小限の `pyproject.toml`

**回答**:
>完全に切り離してください。スキル単位で分離すること。依存関係や管理手段は統一すること（requirements.txt使ってない認識）

---

## 項目 #3 — `user_id` ランダム化 (schema migration)

### 背景
現状の `user_id` は `discord:123456` のようにアダプター名 + プラットフォーム ID
を平文で連結している。DB ダンプが漏洩した場合、Discord/Slack など外部の
アカウントと直接結びついてしまう。

### 想定する変更
- 新規: `user_id = uuid4().hex` のランダム値
- 外部 ID との紐付けは既存の `platform_links` テーブル経由
- 既存 DB の migration スクリプト (`alembic` 相当 or 自前) が必要

### 質問

**Q3-1. 既存データの扱い**
- [ ] A. 自動マイグレーション（起動時に旧 ID を新 UUID に置換、全テーブルを再書き込み）
- [ ] B. 手動 CLI (`cordbeat migrate user-ids`) を用意
- [X] C. 既存 DB は破壊的アップグレード（新規 DB を前提・データ移行は利用者責任）

**Q3-2. 旧 ID の後方互換**
- [ ] A. 旧形式 `discord:xxx` の lookup もサポート（移行期間中）
- [X] B. 即座に廃止（新形式のみ）

**Q3-3. 既存の ChromaDB コレクションメタデータ**
- [ ] A. 起動時に user_id を書き換えて再インデックス
- [X] B. 旧メタデータは破棄（記憶は失われる）
- [ ] C. 別途ツールで変換

**回答**:
>リリースしてないので破壊的変更でよい。即廃止で

---

## 項目 #4 — Forgetting を lazy decay 化

### 背景（詳細）

**現状の動作** (`src/cordbeat/memory.py::decay_and_archive_memories` → `_memory_vector.py::decay_and_archive`):
1. Heartbeat ループが深夜帯に `decay_and_archive_memories()` を呼ぶ
2. `semantic_memory` / `episodic_memory` コレクションから **全件取得**
3. 各エントリについて `calculate_strength(base, elapsed_days, emotion)` を計算
4. 結果を ChromaDB metadata の `decayed_strength` フィールドに **書き戻し**
5. 閾値を下回ったものは **物理削除**

**問題点**:
- 記憶が N 件あると毎晩 O(N) の read + write が発生
- ChromaDB は全件 scan + 全件 upsert が重い（ディスク I/O とインデックス再構築）
- 実際に参照されない古い記憶まで毎晩コストを払う
- Heartbeat のレイテンシが記憶量に比例して悪化

**lazy decay 方式の提案**:
- 書き込みフィールドは `base_strength`, `created_at`, `emotion_weight` のみ（不変）
- `decayed_strength` は **保存しない** — `get_*` / `search_*` の中で毎回計算
- Strength 計算は単純な数式 (Ebbinghaus) なのでコストほぼ 0
- 閾値を下回った記憶は、その記憶が **参照されたタイミング** で初めて削除判定
- 参照されない記憶はそのまま残存（容量的に問題ない想定 — 記憶は今もテキストで軽量）

**トレードオフ**:
| 観点 | 現状 (eager) | 提案 (lazy) |
|------|--------------|-------------|
| 夜間バッチ負荷 | 高（全件） | ほぼ 0 |
| 読み出し時計算 | 無し（metadata 参照） | 軽い算術 1 回 |
| 容量 | 閾値下は即削除 | 参照されるまで残る |
| 実装複雑度 | 中 | 低（state を減らす） |

### 想定する変更
- `_memory_vector.py` の `add_semantic` / `add_episodic` から `decayed_strength` 書き込みを除去
- `search_*` / `get_*` の結果に `calculate_strength()` を適用して返す
- `decay_and_archive` は閾値未満エントリの削除のみ残す（もしくは丸ごと削除）
- `MemoryEntry` モデルから `decayed_strength` を削除 or deprecate

### Q4-1 の選択肢詳細

- **A. 読み出し時に判定し即削除**: 検索結果を返す前に閾値未満を `collection.delete()`。参照のたびに書き込みが入る（read-amplification）
- **B. 読み出し時にマーキングのみ、削除は低頻度バッチ**: 閾値未満を別テーブル / メタデータにマーク、週次 or 月次で一括削除
- **C. 完全に読み出し時計算のみ、物理削除はしない**: 常に全件残す。容量は増え続けるが計算はシンプル

### 質問

**Q4-1. Archive のタイミング**
- [x] A. 読み出し時に判定し即削除（read-amplification あり）
- [ ] B. 読み出し時にマーキングのみ、削除は低頻度バッチ
- [ ] C. 完全に読み出し時計算のみ、物理削除はしない（容量無制限）

**Q4-2. 既存 decay_and_archive バッチ**
- [ ] A. 残す（lazy と両立、デフォルト off）
- [x] B. 廃止（lazy のみ）

**Q4-3. ChromaDB metadata の decayed_strength フィールド**
- [ ] A. 残す（互換性のため）
- [x] B. 削除（base_strength + created_at から常に計算）

**回答**:
>状況や内容がわからないのでもう少し詳細に説明して
>見る感じlazy移行でよさそう（完全置き換え）
---

## 項目 #7 — Chroma → sqlite-vec 置換

### 背景
ChromaDB は別プロセス相当の依存（SQLite + HNSW インデックス）を持ち、
起動コスト・ディスク使用量が大きい。`sqlite-vec` は拡張を読むだけなので
単一 SQLite ファイルに集約できる。

### 想定する変更
- `aiosqlite.connect(...)` → `load_extension('sqlite_vec')`
- `semantic_memory` / `episodic_memory` を同じ DB のテーブルに統合
- Embedding は従来どおり `sentence-transformers` or OpenAI で生成
- `cordbeat/_memory_vector.py` を sqlite-vec 実装で書き換え

### 質問

**Q7-1. 移行戦略**
- [X] A. sqlite-vec に完全置換（ChromaDB 削除）
- [ ] B. 両方サポート（`MemoryConfig.vector_backend = "chroma" | "sqlite-vec"`）
- [ ] C. 保留（ベンチマークを先に取る）

**Q7-2. 既存 ChromaDB データの扱い**
- [ ] A. 起動時に自動インポート（一回限り）
- [ ] B. 手動 CLI で変換
- [X] C. 破壊的（既存データは失われる）

**Q7-3. Embedding 関数**

**背景（詳細）**:
現状、ChromaDB は **組み込みの default embedding** (`all-MiniLM-L6-v2`, sentence-transformers) を
自動で呼び出して内部でベクトル化している。ChromaDB を捨てる場合、embedding は自分で生成して
sqlite-vec に INSERT する必要がある。候補は以下:

- **sentence-transformers (現状継続)**:
  - ✅ 完全ローカル、API コスト 0、プライバシー完全
  - ✅ 初回のみモデル DL（~80MB）、以降はオフライン動作
  - ❌ 推論は CPU で数十ms/件、埋め込み品質は OpenAI 比で劣る
  - ❌ `sentence-transformers` と `torch` が重依存（インストールサイズ大）
- **OpenAI `text-embedding-3-small`**:
  - ✅ 埋め込み品質が高い（多言語、文脈理解が上）
  - ✅ `torch` 依存を落とせる（大幅にスリム化、Docker image が小さくなる）
  - ✅ 1M tokens で $0.02 と安価
  - ❌ API key 必須、オフライン動作不可（cordbeat の "local-first" 方針と矛盾）
  - ❌ ネットワーク依存（レイテンシ、障害時に記憶書き込みが失敗）

**Local-first を崩すかが最大の判断軸です**。

- [x] A. 現状維持（Chroma 組み込み default → sentence-transformers を手動呼び出し）
- [ ] B. 本機会に OpenAI `text-embedding-3-small` へ切替
- [ ] C. ユーザー設定可能（config で選択、デフォルトは sentence-transformers）

**回答**:
>Embeddingに移行するメリットは？
>ローカルファースト思想を守りたい

---

## 項目 #10 — 本物 Chroma (または sqlite-vec) integration test

### 背景
既存テストは `FakeChromaCollection` を使っており、実装差し替え時に
回帰が検知できない。

### 想定する変更
- `tests/integration/test_memory_vector.py` を新設
- `pytest.mark.integration` で分離、CI は `uv run pytest -m integration` を別ジョブ
- 一時ディレクトリに本物の Chroma / sqlite-vec を立てて挿入→検索→decay を検証

### 質問

**Q10-1. CI での実行**
- [X] A. デフォルトジョブに含める（CI 時間増を許容）
- [ ] B. nightly 専用ワークフロー
- [ ] C. ローカルのみ（PR CI では走らせない）

**Q10-2. 対象 backend**
- [ ] A. ChromaDB のみ
- [X] B. sqlite-vec のみ（#7 完了後）
- [ ] C. 両方（#7 で dual-backend を採用した場合）

**Q10-3. カバレッジ閾値への影響**
- [X] A. 現行 78% を維持（integration は coverage 対象外）
- [ ] B. 引き上げ目標（85% など）

**回答**:
>カバレッジは可能なら引き上げたい

> **【Copilot 整合性チェック】** Q10-3 は A ですが、以下 Q10-X-coverage では **C (85% 段階引き上げ)** を選択されています。**最新の Q10-X-coverage=C を優先**し、Q10-3 の選択は無効化して進める認識で OK でしょうか？ 念のため確認:
>
> - [x] OK、Q10-X-coverage=C を最終決定として扱う
> - [ ] いや Q10-3=A を優先（integration 追加はするが閾値は据え置き）
>
> **回答**:
> >

---

## 全体方針

**Q-overall. Batch C の取り組み順**
以下を優先度順に並べてください（不要な項目は除外可）:

- 候補: #7 sqlite-vec / #4 lazy decay / #2 skill venv / #3 user_id ランダム化 / #10 integration test

**回答**:
>推奨順 (#7 / #3 並行 → #4 / #2 並行 → #10) で OK
>
> **Copilot 推奨順序** (依存関係ベース):
>
> | 段 | PR | 理由 |
> |---|----|------|
> | 1 | **#7 sqlite-vec** (破壊的) | memory 全面書き換えの土台。#4/#10 が依存 |
> | 1 (並行) | **#3 user_id ランダム化** (破壊的) | memory subsystem に触れないので #7 と並行 PR 可。Q-overall-2 で B（合流）も選択可 |
> | 2 | **#4 lazy decay** | sqlite-vec 実装に直接組み込む方が簡単 |
> | 2 (並行) | **#2 Skill uv 環境** | memory と独立、並行 PR 可 |
> | 3 | **#10 integration test** | 全ての memory 変更が終わった後に本物 DB で検証 |
>
> **所要 PR 数見積もり**: 5 PR (#3+#7 合流なら 4 PR)
>
> この順序で問題なければ **「推奨順で OK」** とご回答ください。変更希望があれば上の優先度表を直接書き換えてください。

**Q-overall-2. 1 PR 1 項目 原則**
- [ ] A. 維持（各項目を独立 PR）
- [x] B. #3 + #7 は同時に実施（どちらも DB schema を触るため）

**回答**:
>1項目として圧縮できるならまとめてもよいが更新量が多い場合分離

> **【Copilot 補足】** #3 と #7 の合流について精査した結果:
>
> - **#3** は `sqlite` の users / platform_links テーブル (schema) + adapter / gateway レイヤのユーザー解決ロジックを触る
> - **#7** は ChromaDB → sqlite-vec の置換で、memory subsystem の **vector 層だけ** を触る
> - 触る領域が **ほぼ重なりません** (同じ SQLite DB ファイルに書くだけで、テーブルは別)
> - 合流するとレビュー粒度が大きくなりすぎ、回帰時の二分探索も難しくなる
>
> **推奨は分離 (A)**。回答コメント「更新量が多い場合分離」に完全に合致します。
>
> 分離で進めて OK でしょうか？
> - [x] OK、分離 (A) で進める
> - [ ] やはり合流 (B) で

---

## 追加検討事項（ゼロベース見直し）

各項目について、上記の質問では拾いきれていない論点を列挙する。判断が必要なものは
**「要決定」**、報告のみの注意事項は **「FYI」** と表記する。

### #2 Skill subprocess venv — 追加論点

| # | 論点 | 種別 | 内容 |
|---|------|------|------|
| 2-X1 | subprocess 起動コスト | 要決定 | spawn は ~100–300ms/call。pooling（永続ワーカー）にするか、毎回 spawn か |
| 2-X2 | 同時呼び出し制御 | 要決定 | 1 Skill が同時に複数 invoke された場合、venv 単位で lock する？並列許可？ |
| 2-X3 | IPC タイムアウト / クラッシュ処理 | FYI | subprocess hang・crash の検知、kill ポリシー、エラー伝搬の設計が必要 |
| 2-X4 | stdout/stderr の host 側ログ集約 | FYI | Skill 側 print がホスト logger に出るような中継が必要 |
| 2-X5 | venv 配置場所 | 要決定 | `.venv-skills/` をリポジトリ直下？XDG data dir？.gitignore 必須 |
| 2-X6 | Skill の依存更新検出 | 要決定 | `pyproject.toml` のハッシュをキャッシュし、変更時のみ `uv pip install` 再実行する設計 |
| 2-X7 | クロスプラットフォーム | FYI | Windows `Scripts/python.exe` vs Linux `bin/python` の分岐が必要 |
| 2-X8 | 既存 Skill のマイグレーション | 要決定 | 現行 `skills/api_call/`, `skills/web_search/` に `pyproject.toml` を新設する作業が必要（dependency: chromadb はホスト依存？外部 lib なし？） |

**追加質問 Q2-X-pool**: subprocess の起動戦略
- [x] A. 毎回 spawn（実装シンプル、レイテンシ ~200ms/call）
- [ ] B. Skill ごとに常駐 worker pool（高速、複雑度↑）
- [ ] C. まず A で実装、ホットパスのみ B に最適化

**回答**:
>何度か行っていますが仮想化方法は統一してね。複数種類使わない（uv使ってるのでvenv使わない）

> **【Copilot 補足】** 承知しました。提案の `uv venv .venv-skills/<name>/` は内部的には venv 互換ディレクトリを作る `uv` のサブコマンドで、**ツールは `uv` で統一**されます（`python -m venv` や `virtualenv` のような別ツールは使いません）。依存インストールも `uv pip install` 経由です。命名上 "venv" と呼んでいるだけで、実体は uv が管理する Python 環境です。これで認識あっていますか？ 念のため確認用の追加質問を下記に追加します。
>
> **確認質問 Q2-X-tooling**: 「uv 管理環境」という呼び方
> - [x] A. OK、`.venv-skills/` のままでよい
> - [ ] B. `.uv-skills/` にリネームして混乱を避ける
> - [ ] C. ドキュメント上で「uv が作った独立 Python 環境」と明記すれば名前は問わない
>
> **回答**:
> >

### #3 user_id ランダム化 — 追加論点

| # | 論点 | 種別 | 内容 |
|---|------|------|------|
| 3-X1 | ID 形式 | 要決定 | `uuid4().hex` (32 char) / `ulid` (sortable) / `secrets.token_urlsafe(16)` のどれか |
| 3-X2 | ログ出力での扱い | FYI | 新 ID も結局ログに出る。漏洩時のリスクは「外部アカウントとの紐付けが切れる」点だけ |
| 3-X3 | テストフィクスチャ更新 | FYI | `tests/` 内の `discord:test_user` 等のハードコーディング箇所をランダム ID 化 |
| 3-X4 | 1 user が複数 platform | FYI | link token フローはそのまま、`platform_links` で N:1 ぶら下げる設計でカバー済 |
| 3-X5 | `cordbeat status` 等の CLI 表示 | 要決定 | 内部 ID をユーザーに見せるか？display_name のみ？ |

**追加質問 Q3-X-format**: ID 形式の選択
- [x] A. `uuid4().hex` (32文字、衝突確率実質ゼロ)
- [ ] B. `ulid` (26文字、時系列ソート可能、追加依存)
- [ ] C. `secrets.token_urlsafe(16)` (22文字、URL safe)

**回答**:
>安全性は重視してください

### #4 lazy decay — 追加論点

| # | 論点 | 種別 | 内容 |
|---|------|------|------|
| 4-X1 | 検索結果のソート | 要決定 | 現状はベクトル類似度順？ decayed_strength 順？ lazy 化後は post-fetch で strength を掛けて再ソートが必要 |
| 4-X2 | 検索時の閾値フィルタ | 要決定 | 強度が低いエントリを「削除はしないが結果から除外」する閾値も併設するか |
| 4-X3 | 既存データの decayed_strength 残骸 | FYI | 過去エントリには metadata 残置。読み出し時は無視で OK（ユーザー回答 #3 で破壊的変更 OK のため不問） |
| 4-X4 | reflection / dream layer での参照 | FYI | heartbeat の reflection が「強い記憶を引く」処理に依存していないか確認必要 |

**追加質問 Q4-X-sort**: 検索結果のソート
- [ ] A. ベクトル類似度のみ（strength は表示用）
- [x] B. similarity × strength の合成スコア（古い記憶は順位が下がる）
- [ ] C. similarity でフィルタ → strength で再ソート

**回答**:
>思想にあったのはどれ？よくわからない

> **【Copilot 補足】** CordBeat は「人間らしい記憶」を模しており、Ebbinghaus の忘却曲線を採用している点からも **「古い記憶は優先順位が下がる」** 方針が自然です。
>
> - **A (類似度のみ)**: 最も似ている記憶を機械的に返す。古くてもヒットする = 「最近のこと」より「昔の強烈な出来事」が優先される可能性あり → 人間らしくない
> - **B (similarity × strength の合成)**: 「似ていて、かつ鮮明に覚えている」記憶が優先される。**人間の想起に最も近い** (似た状況でも鮮度の高い方を思い出す)
> - **C (類似度でフィルタ → 強度で再ソート)**: 候補を類似度で絞り込んでから強度順に並べ替え。B に近いがロジックが分離
>
> **思想的推奨は B**。ただし実装は似たり寄ったりで、B / C どちらでも同じ最終結果に近づきます。B が最も自然です。

### #7 sqlite-vec — 追加論点

| # | 論点 | 種別 | 内容 |
|---|------|------|------|
| 7-X1 | SQLite 拡張ロード可否 | 要確認 | システム Python の SQLite が `enable_load_extension` を許可していない場合あり。aiosqlite 経由で確認が必要 |
| 7-X2 | ベクトル次元固定 | FYI | sentence-transformers MiniLM = 384 次元。テーブル作成時に固定 |
| 7-X3 | インデックス種別 | 要決定 | sqlite-vec の `vec0` (brute force) で十分か、HNSW が必要か（記憶数の想定次第） |
| 7-X4 | metadata フィルタ | FYI | sqlite-vec はベクトル列のみ持つテーブル。user_id フィルタは別テーブル JOIN |
| 7-X5 | テスト用 in-memory DB | 要決定 | 既存 FakeChromaCollection 相当を作るか、本物の sqlite-vec を使うか |
| 7-X6 | ベンチマーク | FYI | 切替前後でクエリレイテンシ計測を README に残しておくと後で安心 |

**追加質問 Q7-X-index**: インデックス戦略
- [x] A. 当面 brute force（数千件想定）
- [ ] B. 最初から HNSW（実装コスト↑だがスケーラブル）
- [ ] C. 設定で切替可能

**回答**:
>なにが問題かわからん

> **【Copilot 補足】** 「ベクトル検索をどうやって速くするか」の話です。
>
> - **A. Brute force (全件計算)**: 検索のたびに全記憶とコサイン類似度を計算。記憶が数千件なら瞬時 (~10ms)、数万件だと目に見えて遅く (~100ms〜)、数十万件で体感悪化。実装が最も単純
> - **B. HNSW**: 近似近傍探索アルゴリズム。大量データでも一定速度を保てる (~数ms)。実装はやや複雑、sqlite-vec の最新版が必要、ディスク使用量も増
> - **C. 設定で切替**: 利用者環境に応じて変えられる。柔軟だが二重実装コスト
>
> **判断基準**: CordBeat は個人用ローカル AI なので、1 ユーザーあたりの記憶は数千〜せいぜい 1〜2 万件想定。**A で十分**と思われます。B/C は将来 10 万件超えてから導入でも遅くない。
>
> **推奨: A** (シンプル + 実害なし、必要になったら後から HNSW に差し替え可能)

### #10 integration test — 追加論点

| # | 論点 | 種別 | 内容 |
|---|------|------|------|
| 10-X1 | カバレッジ閾値の方針整理 | 要決定 | Q10-3 で「78% 維持」を選択しつつ「可能なら引き上げ」と回答あり。整合: integration を coverage 対象に含め、新閾値（例: 80%）を設定するのが筋 |
| 10-X2 | CI 時間増の見積り | FYI | sqlite-vec + 実 embedding (sentence-transformers) で初回モデル DL ~80MB、CI キャッシュ必須 |
| 10-X3 | flaky 対策 | FYI | embedding は決定論的だが浮動小数点誤差で similarity 順位が稀にブレる。距離マージン付きで assert する |

**追加質問 Q10-X-coverage**: カバレッジ閾値の最終形
- [ ] A. 78% 据え置き、integration は coverage 対象外
- [ ] B. integration を含めて測定し、80% へ引き上げ
- [x] C. integration を含めて測定し、85% を目指す（達成までは段階的に）

**回答**:
>CIの時間増については問題ですか？

> **【Copilot 補足】** 問題かどうかの判断材料:
>
> **現状の CI 時間** (PR #78 実測):
> - Ubuntu 3 ジョブ: 各 ~34 秒
> - Windows 3 ジョブ: 各 2〜3 分
> - 全体完了まで並列で約 3〜4 分
>
> **integration test を含めた場合の見込み**:
> - sentence-transformers モデル DL (初回 ~80MB、~20 秒) → GHA キャッシュで 2 回目以降は ~3 秒
> - sqlite-vec で数百件 INSERT + 検索 → ~10 秒
> - **トータル: 各ジョブ +30〜60 秒程度**
> - GHA 無料枠 (月 2,000 分) に対する影響は小さく、コスト面は問題なし
>
> **結論**: CI 時間増は **問題になりません**。integration を含めて測定し、閾値を引き上げる **B or C** が筋です。
>
> - 段階的引き上げが安全なので **C (85% を目指す、達成まで段階的)** を推奨
> - 急に引き上げるとテスト書き足しが間に合わず PR が通らなくなるリスクあり

### 全体方針 — 追加論点

| # | 論点 | 種別 | 内容 |
|---|------|------|------|
| O-X1 | 依存関係 | 要決定 | #4 lazy decay と #10 integration test は #7 (sqlite-vec) 完了後の方が実装が単純。`#7 → #4 → #10` を推奨 |
| O-X2 | 並行作業可能 | FYI | #2 (skill venv) と #3 (user_id) は memory subsystem に触らないので #7 系列と並行 PR 可 |
| O-X3 | CHANGELOG | FYI | #3 + #7 は破壊的変更。`BREAKING:` プレフィクスを付けて記録 |
| O-X4 | バージョニング | 要決定 | 破壊的変更を含む PR をマージしたら `0.x` → `0.(x+1)` の minor bump にするか、`v1` 直前の整理として major へ進めるか |
| O-X5 | ドキュメント更新 | FYI | `README.md` / `docs/` の memory subsystem 説明、Skill 開発ガイド、migration note の追記が必要 |

**追加質問 Q-O-X-version**: バージョン番号方針
- [x] A. 各 PR ごとに minor bump（0.x.0 → 0.(x+1).0）
- [ ] B. Batch C 全部を 1 つの release にまとめて 1.0 へ昇格
- [ ] C. その他（自由記述）

**回答**:
>PRした後にこれを1.0にするとできる？破壊的更新ができなくなると思ってるのでできればギリギリまで粘りたい

> **【Copilot 補足】** 結論: **`0.x` のままで問題ありません。1.0 昇格は任意のタイミングで可能**です。
>
> **SemVer の考え方**:
> - `0.x.y` は "公開 API が不安定" の明示的な合図。破壊的変更は `0.x` → `0.(x+1)` で自由に行える
> - `1.0.0` 以降は破壊的変更で必ず `2.0.0` に bump する必要がある (ユーザーとの契約)
>
> **推奨フロー**:
> 1. Batch C の破壊的 PR (`#3`, `#7`) は `0.(x+1).0` の minor bump で入れる
> 2. その後しばらく使ってみて「この API で安定した」と判断できたら `1.0.0` へ昇格
> 3. 1.0 以降は破壊的変更を控える (どうしても必要なら `2.0` へ)
>
> **つまり**: Batch C 作業中は `0.x` の minor bump を続けるのが安全。1.0 は Batch C 後の「落ち着いた」タイミングで決める。「ギリギリまで粘る」戦略と完全に一致します。
>
> - 推奨選択肢: **A (各 PR で minor bump)**

---

## サマリ — まだ未回答の項目

優先回答お願いしたい欄:
- **#4 Q4-1 / Q4-2 / Q4-3**（決定が決まれば実装に進める）
- **#4 Q4-X-sort** — 推奨: **B** (similarity × strength の合成、人間の想起に最も近い)
- **#7 Q7-3** — 推奨: **A** (local-first 維持)
- **#7 Q7-X-index** — 推奨: **A** (brute force で十分、将来 HNSW に差替可能)
- **#10 Q10-X-coverage** — 推奨: **C** (integration を含め 85% へ段階的引き上げ、CI 時間増は問題なし)
- **Q-O-X-version** — 推奨: **A** (各 PR で minor bump、1.0 は Batch C 後に決定)
- **#2 Q2-X-pool** — 推奨: **C** (まず spawn、ホットパスのみ後で最適化)
- **#2 Q2-X-tooling** (新規追加) — `.venv-skills/` 命名の確認
- **#3 Q3-X-format** (すでに A 選択済み、回答「安全性重視」) — `uuid4` でそのまま進めて OK です
- **Q-overall（順序）** — 推奨: `#7 → #4 → #10` を memory 系列、`#2`, `#3` を並行枠
