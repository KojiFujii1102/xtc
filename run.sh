#!/bin/bash

# MNIST Deep Learning モデル訓練・評価スクリプト

echo "======================================"
echo "MNIST Deep Learning モデル"
echo "======================================"
echo ""

# 依存パッケージのチェック
echo "依存パッケージをインストール中..."
pip install -q -r requirements.txt
echo "✓ インストール完了"
echo ""

# モデルの訓練
echo "======================================"
echo "1. モデルの訓練"
echo "======================================"
python src/train_mnist.py
echo ""

# モデルの評価
echo "======================================"
echo "2. モデルの評価"
echo "======================================"
python src/predict_mnist.py
echo ""

echo "======================================"
echo "完了!"
echo "======================================"
echo ""
echo "次のステップ:"
echo "  - 訓練履歴: training_history.png"
echo "  - 予測結果: predictions.png"
echo "  - 混同行列: confusion_matrix.png"
echo "  - 信頼度分布: confidence_distribution.png"
echo ""
echo "Webアプリを起動するには:"
echo "  streamlit run src/app.py"
