[English](README.md) | [Tiếng Việt](README.vi.md) | **日本語**

# tsubuyaki

字幕の学習フローを保ちながら、文脈に応じた編集可能なふりがなを生成する
日本語メディアリーダーです。

[公開アプリを開く](https://voxnuts947-tsubuyaki.hf.space) ·
[Hugging Face Space](https://huggingface.co/spaces/voxnuts947/tsubuyaki) ·
[Kaggle ノートブック](https://www.kaggle.com/code/voxnuts465/furigana-aid-generation)

## スクリーンショット

### メディアライブラリ

![tsubuyaki のメディアライブラリ](https://raw.githubusercontent.com/VoxNut/tsubuyaki/main/demo%20images/home%20page.png)

### 文脈対応ふりがな付きリーダー

![tsubuyaki のリーダー](https://raw.githubusercontent.com/VoxNut/tsubuyaki/main/demo%20images/main%20page.png)

## 主な変更点

kikiyomi のリスニング体験をもとに、ふりがな学習を支援する機能を追加
しました。

- ファインチューニング済み Japanese BERT と target-aware MLP による
  文脈依存の読み分け。
- surface ごとの候補マスクと読み prior により、不可能な予測を除外。
- validation で調整した BERT–MLP ensemble と、Most-Frequent／MeCab
  fallback。
- ふりがなを漢字部分だけに配置し、送り仮名は ruby の外側に保持。
- `Always` と `On hover` の表示モード、読みサイズ調整、手動編集。
- アプリ独自 JSON を round-trip の正本とし、ruby タグ付き SRT も共有・
  再インポート用に出力可能。
- Rosé Pine パレットと統一した inline SVG アイコンによる英語 UI。
- 1 つの Docker image から静的 frontend と FastAPI の両方を配信。

## 機能

### リーダー

- 動画、音声、SRT、アプリ独自 JSON のドラッグ＆ドロップ。
- 動画／音声 player、字幕への seek、replay、Focus mode、Chapters、
  Bookmarks、履歴、学習進捗。
- キーボード、gamepad、desktop、mobile に対応。
- MP4、WebM、MKV、MOV、M4V、OGV、AVI、MPEG/MPG、MP3、M4A/M4B、
  AAC、OGG/OGA、OPUS、WAV、FLAC を選択可能。
- 実際の再生可否はブラウザの codec 対応に依存します。MP4 H.264/AAC と
  WebM VP9/Opus が最も安全です。

### ふりがな

- すべての字幕 cue を一括処理し、進捗表示とキャンセルに対応。
- 読みを日本語テキストの上に配置し、漢字と送り仮名を正しく分離。
- 各 reading の confidence と prediction source を JSON に保存。
- ruby segment をクリックして読みを編集または削除。
- アプリ独自 JSON と ruby タグ付き `.furigana.srt` を出力。

### Round-trip JSON

JSON には次の情報が保持されます。

- 各 cue の timestamp と plain text
- ruby segment と reading
- confidence と prediction source
- 手動編集
- model と artifact の version

HTML は表示形式であり、アプリの主要データ構造ではありません。

## 推論パイプライン

1. MeCab が日本語字幕の各 cue を tokenize します。
2. 既知の surface を許可された読み候補に対応付けます。
3. 複数の読みを持つ surface を文中で `[TGT]` と `[/TGT]` により示します。
4. Japanese BERT が文脈表現と classifier logits を生成します。
5. Target-aware MLP が CLS、target marker、target span の特徴を使用します。
6. 候補マスク適用後の BERT／MLP score と surface log prior を統合します。
7. Confidence が低い場合は train 内の mode reading に fallback します。
8. 文脈を必要としない surface や未知語には Most-Frequent または MeCab
   fallback を使用します。
9. 選択した reading を漢字部分に対応付け、構造化 JSON segment として返し
   ます。

API が返す prediction source は `ContextEnsemble`、
`MostFrequentLowConfidence`、`MostFrequent`、`MeCabFallback`、`PlainText`、
`ManualEdit` です。

## データと再現性

学習ノートブックは
[`Calvin-Xu/Furigana-Aozora`](https://huggingface.co/datasets/Calvin-Xu/Furigana-Aozora)
を streaming で読み込みます。同じ source file の例が train、validation、
test をまたがないよう、file 単位で group split しています。

| 項目 | 値 |
| --- | ---: |
| Random seed | 42 |
| 全 target 数 | 259,476 |
| Train / validation / test | 207,635 / 25,914 / 25,927 |
| Train surface 数 | 26,210 |
| 有効な曖昧 surface 数 | 1,004 |
| Reading label 数 | 1,491 |
| 最大 sequence length | 192 |
| Target-aware MLP validation accuracy | 95.20% |
| Ensemble validation accuracy | 95.46% |
| 調整後 validation accuracy | 95.48% |

デプロイ済み model repository は private で、変更不能な commit に pin して
います。API が ready になる前に、artifact checksum、label mapping、tokenizer
marker、metadata を検証します。

## アーキテクチャ

| コンポーネント | 役割 |
| --- | --- |
| `frontend/` | メディアライブラリ、player、字幕、ふりがな編集、export |
| FastAPI | 同一 origin から frontend と `/api` endpoint を配信 |
| BERT classifier | 文脈に応じた reading score |
| Target-aware MLP | 対象語に特化した文脈特徴と reading score |
| Candidate artifacts | Surface mask、prior、mode、label、calibration |
| MeCab | Tokenization と fallback reading |

Frontend と backend を同一 origin で配信するため、デプロイが単純になり、
別の CORS 境界も不要です。

## Docker でローカル実行

CPU image を build します。

```bash
docker build -t tsubuyaki .
```

ローカルの model directory を使って起動します。

```bash
docker run --rm -p 8000:7860 \
  -e FURIGANA_MODEL_LOCAL_DIR=/model \
  -e FURIGANA_DEVICE=cpu \
  -v /absolute/path/to/model:/model:ro \
  tsubuyaki
```

`http://localhost:8000/` を開いてください。

Hugging Face の private model を使用する場合は、次の値を環境変数または
デプロイ先の secret として設定します。

```text
FURIGANA_HF_MODEL_REPO=owner/private-model
FURIGANA_HF_MODEL_REVISION=<immutable-40-character-commit-sha>
HF_TOKEN=<dedicated-read-only-token>
FURIGANA_DEVICE=cpu
```

Access token を commit したり、Docker image に含めたりしないでください。

## API

| Method | Endpoint | 用途 |
| --- | --- | --- |
| `GET` | `/api/health` | Process の health check |
| `GET` | `/api/ready` | Model と artifact の readiness |
| `GET` | `/api/version` | API、model、artifact の version |
| `POST` | `/api/furigana/generate-batch` | 字幕 cue の構造化ふりがな生成 |

## デプロイ

公開版は Hugging Face Docker Space の CPU Basic で稼働しています。

<https://voxnuts947-tsubuyaki.hf.space>

同じ Dockerfile を VPS や GPU host でも使用できます。
`FURIGANA_DEVICE=auto` は、対応 GPU と CUDA-enabled PyTorch build の両方が
利用できる場合にのみ CUDA を選択します。

[英語のデプロイガイド](docs/DEPLOY_EN.md)または
[ベトナム語のデプロイガイド](docs/DEPLOY.md)も参照してください。

## 関連ドキュメント

- [アーキテクチャ](docs/ARCHITECTURE.md)
- [Phase 1 実装ノート](docs/PHASE1.md)
- [セキュリティと secret の管理](docs/SECURITY.md)
- [Model の制限事項](docs/MODEL_LIMITATIONS.md)

## 制限事項

- 人名、珍しい語、専門分野の文章、未知の surface では、自動 reading が誤る
  ことがあります。必要に応じて reading editor で修正してください。
- CPU の cold start には private model の download と検証が含まれます。
- ブラウザが container を認識しても、内部の全 codec を decode できるとは
  限りません。
- Confidence は model の指標であり、言語的な正しさを保証するものでは
  ありません。

## 帰属とライセンス

Player のワークフローは
[`rtr46/kikiyomi`](https://github.com/rtr46/kikiyomi)をもとにしています。
帰属情報は [NOTICE.md](NOTICE.md) に保持され、本プロジェクトは
[GPL-3.0](LICENSE) で公開されています。

UI には公式の
[Rosé Pine palette](https://github.com/rose-pine/rose-pine-palette)を使用して
います。
