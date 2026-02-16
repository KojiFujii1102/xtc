# XTC - 機械学習プロジェクト集

## プロジェクト一覧

このリポジトリには複数の機械学習プロジェクトが含まれています：

### 1. 🤖 シンプルLLMチャットボット

Claude APIを使用したシンプルなチャットボットアプリケーションです。

**機能：**
- 💬 リアルタイムストリーミングチャット
- 🎛️ モデル選択（Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opusなど）
- 🌡️ Temperature調整
- 📝 チャット履歴の保持とクリア
- 🎨 シンプルで使いやすいUI

**使い方：**
```bash
streamlit run chatbot.py
```

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
├── chatbot.py            # LLMチャットボット
└── requirements.txt      # Python依存パッケージ
```

## 🚀 セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. プロジェクトの選択

#### LLMチャットボットを使う場合

Claude API Keyが必要です。`.env`ファイルを作成するか、アプリのサイドバーから入力してください。

```bash
streamlit run chatbot.py
```

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

### LLMチャットボット
- Python 3.x
- Streamlit
- Claude API (Anthropic)

### MNIST分類
- TensorFlow 2.13以上
- Keras 2.13以上
- NumPy, Matplotlib, Seaborn
- scikit-learn, Pillow

## 📝 ライセンス

MIT License
