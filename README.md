# XTC - 機械学習プロジェクト集

## プロジェクト一覧

このリポジトリには複数の機械学習プロジェクトが含まれています：

### 1. 🤖 Claude チャットボット

Claude APIを使用したチャットボットです。コンソール版とStreamlit版の2種類があります。

#### 💻 コンソール版 (`claude_chat.py`)

**機能：**
- 💬 コンソールでの対話型チャット
- 🔄 会話履歴の保持
- 📝 ストリーミングレスポンス
- 🎯 シンプルで使いやすい

**使い方：**
```bash
# API Keyを環境変数に設定（推奨）
export ANTHROPIC_API_KEY="your-api-key-here"

# チャットボットを起動
python3 claude_chat.py
```

終了するには `exit`、`quit`、または `q` と入力してください。

#### 🌐 Streamlit版 (`streamlit_chat.py`)

**機能：**
- 🖥️ ブラウザベースのWebインターフェース
- 💬 リアルタイムストリーミングチャット
- 🎛️ モデル選択（Claude 3.5 Sonnet, Haiku, Opus）
- 🔧 Max Tokens設定
- 📝 チャット履歴の保持とクリア

**使い方：**
```bash
# Streamlitアプリを起動
streamlit run streamlit_chat.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。
サイドバーでAPI Keyを入力してください。

### 2. 📊 MNIST手書き数字分類

TensorFlow/Kerasを使用したMNIST手書き数字分類のDeep Learningプロジェクトです。

**機能：**
- CNNアーキテクチャを使用した高精度な分類モデル
- 詳細な評価指標と可視化
- Streamlitを使用したインタラクティブなデモ
- 訓練済みモデルを使用した推論

**プロジェクト構造：**
```
webapp/
├── src/
│   ├── train_mnist.py      # モデル訓練スクリプト
│   ├── predict_mnist.py    # 予測・評価スクリプト
│   └── app.py             # Streamlit Webアプリ
├── models/                # 訓練済みモデル保存先
├── data/                  # データセット保存先
├── logs/                  # TensorBoard ログ
├── claude_chat.py         # Claudeチャットボット（コンソール版）
├── streamlit_chat.py      # Claudeチャットボット（Streamlit版）
├── demo_chatbot.py        # デモンストレーション
└── requirements.txt       # Python依存パッケージ
```

## 🚀 セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. プロジェクトの選択

#### Claudeチャットボットを使う場合

Claude API Keyが必要です。[Anthropic Console](https://console.anthropic.com/)でAPI Keyを取得してください。

**コンソール版:**
```bash
# 環境変数に設定（推奨）
export ANTHROPIC_API_KEY="your-api-key-here"

# チャットボットを起動
python3 claude_chat.py
```

**Streamlit版（Web UI）:**
```bash
# Streamlitアプリを起動
streamlit run streamlit_chat.py
```

ブラウザが自動的に開きます。サイドバーでAPI Keyを入力してください。

#### MNIST分類を使う場合

モデルを訓練：
```bash
python src/train_mnist.py
```

Webアプリを起動：
```bash
streamlit run src/app.py
```

## 🧠 技術スタック

### Claudeチャットボット
- Python 3.x
- Claude API (Anthropic)
- コンソール版 / Streamlit Web版

### MNIST分類
- TensorFlow 2.13以上
- Keras 2.13以上
- NumPy, Matplotlib, Seaborn
- scikit-learn, Pillow

## 📝 使用例

### Claude チャットボット
```bash
$ python3 claude_chat.py

==================================================
Claude チャットボット (コンソール版)
==================================================

チャットを開始します。終了するには 'exit' または 'quit' と入力してください。
--------------------------------------------------

あなた: こんにちは！
Claude: こんにちは！お手伝いできることがあれば教えてください。

あなた: Pythonについて教えて
Claude: Pythonは...

あなた: exit
チャットを終了します。
```

## 📝 ライセンス

MIT License
