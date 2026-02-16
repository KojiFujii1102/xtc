# MNIST Deep Learning モデル - クイックスタートガイド

## 🚀 すぐに始める

### オプション1: すべてを自動実行

```bash
./run.sh
```

このスクリプトは自動的に：
1. 依存パッケージをインストール
2. モデルを訓練
3. モデルを評価
4. 各種グラフを生成

### オプション2: ステップバイステップ

#### 1. 環境セットアップ
```bash
pip install -r requirements.txt
```

#### 2. モデル訓練（10-15分）
```bash
python src/train_mnist.py
```

#### 3. モデル評価
```bash
python src/predict_mnist.py
```

#### 4. Webアプリ起動
```bash
streamlit run src/app.py
```

## 📊 生成されるファイル

- `models/mnist_model.h5` - 訓練済みモデル（H5形式）
- `models/mnist_model.tflite` - 訓練済みモデル（TFLite形式）
- `training_history.png` - 訓練履歴グラフ
- `predictions.png` - 予測サンプル画像
- `confusion_matrix.png` - 混同行列
- `confidence_distribution.png` - 信頼度分布

## 🎯 期待される結果

- **テスト精度**: 99%以上
- **訓練時間**: CPU 10-15分 / GPU 3-5分
- **モデルサイズ**: 約5MB

## 💡 ヒント

- GPUがある場合、自動的に使用されます
- 訓練中はTensorBoardでメトリクスを確認できます：
  ```bash
  tensorboard --logdir logs/fit
  ```
- Webアプリでは独自の手書き数字画像をテストできます

## ❓ トラブルシューティング

### ImportError: No module named 'tensorflow'
```bash
pip install tensorflow
```

### メモリエラー
バッチサイズを減らしてください（`src/train_mnist.py`の`batch_size=128`を`64`に変更）

### Webアプリが起動しない
Streamlitがインストールされているか確認：
```bash
pip install streamlit
```
