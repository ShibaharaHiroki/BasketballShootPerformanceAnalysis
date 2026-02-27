# BasketballShootPerformanceAnalysis

バスケットボールのシュートパフォーマンスを可視化・分析するアプリケーションである。 選手やチームのシュートパフォーマンスを対話的に比較分析できる。

## 技術スタック

### バックエンド

| カテゴリ | ライブラリ |
|---------|-----------|
| Web フレームワーク | FastAPI 0.104 + Uvicorn |
| データ処理 | NumPy, Pandas, openpyxl |
| 分析・次元削減 | TULCA, PaCMAP, scikit-learn, SciPy |

### フロントエンド

| カテゴリ | ライブラリ |
|---------|-----------|
| UI フレームワーク | React 18 + TypeScript |
| コンポーネント | Chakra UI v2 |
| グラフ描画 | Plotly.js / react-plotly.js |
| HTTP 通信 | Axios |
| アニメーション | Framer Motion |

## 前提条件

- **Python** 3.10 以上
- **Node.js** 18 以上（npm 含む）
- **B.League データファイル**（Excel 形式）  
  デフォルトのパスは `D:\data\3.1_イベントデータ(座標付き)_三遠2022-23_2023-24シーズン.xlsx` である。  
  変更する場合は `backend/core/bleague_data_loader.py` 内の `DATA_FILE` を要編集。

## セットアップと実行方法

### バックエンド

```bash
cd backend

# 仮想環境の作成（初回のみ）
python -m venv venv

# 仮想環境の有効化
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt

# サーバー起動
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

起動後、以下の URL でアクセスできる:

- API エンドポイント: http://localhost:8000
- API ドキュメント (Swagger UI): http://localhost:8000/docs

### フロントエンド

```bash
cd frontend

# 依存パッケージのインストール（初回のみ）
npm install

# 開発サーバー起動
npm start
```

起動後、ブラウザで http://localhost:3000 にアクセスする。

> [!TIP]
> プロジェクトルートに用意されたバッチファイルを使うと、上記の手順をワンクリックで実行できる。
> - `start_backend.bat` — 仮想環境の作成・依存インストール・バックエンド起動を一括実行
> - `start_frontend.bat` — 依存インストール・フロントエンド起動を一括実行

## プロジェクト構成

```
BasketballShootPerformanceAnalysis/
├── backend/
│   ├── main.py                  # FastAPI エントリーポイント
│   ├── models.py                # Pydantic スキーマ定義
│   ├── requirements.txt         # Python 依存パッケージ
│   ├── api/
│   │   └── routes.py            # API ルーティング
│   └── core/
│       ├── analysis.py          # TULCA / PaCMAP / RF 分析ロジック
│       ├── aggregations.py      # 集計処理
│       ├── data_loader.py       # データ読み込み（汎用）
│       └── bleague_data_loader.py  # B.League 専用データローダー
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── App.tsx              # アプリケーションルート
│       ├── index.tsx            # エントリーポイント
│       ├── components/
│       │   ├── Sidebar.tsx          # サイドバー（パラメータ設定）
│       │   ├── SpatialHeatmap.tsx   # シュートヒートマップ
│       │   ├── ScatterPlot.tsx      # 2次元散布図
│       │   ├── DominanceMap.tsx     # Dominance Map
│       │   └── RawDataExplorer.tsx  # 詳細データ閲覧
│       ├── context/             # React Context（状態管理）
│       ├── services/            # API 呼び出し
│       └── types/               # TypeScript 型定義
├── start_backend.bat            # バックエンド起動スクリプト
├── start_frontend.bat           # フロントエンド起動スクリプト
└── README.md
```