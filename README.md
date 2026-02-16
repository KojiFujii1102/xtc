# MNIST手書き数字分類 Deep Learning モデル

TensorFlow/Kerasを使用したMNIST手書き数字分類のDeep Learningプロジェクトです。畳み込みニューラルネットワーク（CNN）を使用して、0-9の手書き数字を高精度で分類します。

## 📋 プロジェクト概要

このプロジェクトは以下の機能を提供します：

- **モデル訓練**: CNNアーキテクチャを使用した高精度な分類モデル
- **モデル評価**: 詳細な評価指標と可視化
- **Webインターフェース**: Streamlitを使用したインタラクティブなデモ
- **予測機能**: 訓練済みモデルを使用した推論

## 🏗️ プロジェクト構造

```
webapp/
├── src/
│   ├── train_mnist.py      # モデル訓練スクリプト
│   ├── predict_mnist.py    # 予測・評価スクリプト
│   └── app.py             # Streamlit Webアプリ
├── models/                # 訓練済みモデル保存先
├── data/                  # データセット保存先（自動ダウンロード）
├── logs/                  # TensorBoard ログ
├── requirements.txt       # Python依存パッケージ
└── README.md             # このファイル
```

## 🚀 セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. 必要なパッケージ

- TensorFlow 2.13以上
- Keras 2.13以上
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Pillow
- Streamlit

## 📖 使い方

### モデルの訓練

MNISTデータセットを使用してCNNモデルを訓練します：

```bash
python src/train_mnist.py
```

**訓練の特徴：**
- エポック数: 50（Early Stoppingあり）
- バッチサイズ: 128
- オプティマイザー: Adam (learning_rate=0.001)
- データ拡張: 回転、ズーム、平行移動
- 検証データ分割: 10%
- モデル保存: `models/mnist_model.h5`
- TensorFlow Lite形式も保存: `models/mnist_model.tflite`

**出力：**
- 訓練済みモデル（H5形式、TFLite形式）
- 訓練履歴グラフ（`training_history.png`）
- TensorBoardログ（`logs/fit/`）

### モデルの評価と予測

訓練済みモデルを評価し、詳細な分析を行います：

```bash
python src/predict_mnist.py
```

**出力：**
- テスト精度とロス
- 予測サンプル画像（`predictions.png`）
- 混同行列（`confusion_matrix.png`）
- 信頼度分布（`confidence_distribution.png`）
- 詳細な分類レポート

### Webアプリケーションの起動

Streamlitを使用したインタラクティブなWebインターフェース：

```bash
streamlit run src/app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

**機能：**
- 画像ファイルのアップロード（PNG、JPG、JPEG）
- MNISTサンプルデータでのテスト
- リアルタイム予測
- 予測確率の可視化
- 信頼度メーター

## 🧠 モデルアーキテクチャ

### CNNの構成

```
入力: 28x28x1 (グレースケール画像)

Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(10, softmax)

出力: 10クラス (0-9の数字)
```

**主な特徴：**
- 3層の畳み込みブロック
- Batch Normalizationによる学習安定化
- Dropoutによる過学習防止
- Adam Optimizerによる最適化

## 📊 期待される性能

- **テスト精度**: 99%以上
- **訓練時間**: 約10-15分（CPU）/ 約3-5分（GPU）
- **モデルサイズ**: 約5MB（H5形式）

## 🔧 カスタマイズ

### ハイパーパラメータの調整

`src/train_mnist.py` の以下のパラメータを変更できます：

```python
# モデル構築
classifier = MNISTClassifier()
classifier.build_model()
classifier.compile_model(learning_rate=0.001)  # 学習率

# 訓練
classifier.train(
    x_train, y_train,
    x_val, y_val,
    epochs=50,        # エポック数
    batch_size=128,   # バッチサイズ
    model_path='models/mnist_model.h5'
)
```

### モデルアーキテクチャの変更

`MNISTClassifier.build_model()` メソッドを編集して、層の数、フィルター数、活性化関数などを変更できます。

## 📈 TensorBoardでの可視化

訓練中のメトリクスをTensorBoardで確認：

```bash
tensorboard --logdir logs/fit
```

ブラウザで `http://localhost:6006` にアクセスしてください。

## 🧪 テスト

モデルの性能をテストデータで評価：

```bash
python src/predict_mnist.py
```

詳細な分類レポート、混同行列、信頼度分布が生成されます。

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 📚 参考資料

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🎯 次のステップ

- [ ] データ拡張の強化
- [ ] アンサンブル学習の実装
- [ ] モデルの軽量化（量子化）
- [ ] REST APIの追加
- [ ] Dockerコンテナ化
- [ ] クラウドデプロイメント

## 📞 お問い合わせ

質問や提案がある場合は、issueを作成してください。
