#!/usr/bin/env python3
"""
MNIST文字分類 予測スクリプト

訓練済みモデルを使用して、MNISTデータセットの予測を行います。
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class MNISTPredictor:
    """MNIST文字分類の予測クラス"""
    
    def __init__(self, model_path='models/mnist_model.h5'):
        """
        初期化
        
        Args:
            model_path: 訓練済みモデルのパス
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """訓練済みモデルを読み込む"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルが見つかりません: {self.model_path}")
        
        print(f"モデル読み込み中: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("モデル読み込み完了")
        
    def predict(self, x):
        """
        予測を実行
        
        Args:
            x: 入力データ
            
        Returns:
            予測結果
        """
        predictions = self.model.predict(x, verbose=0)
        return predictions
    
    def predict_single(self, image):
        """
        単一画像の予測
        
        Args:
            image: 28x28の画像データ
            
        Returns:
            予測されたクラスと確率
        """
        # 画像の形状を調整
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = np.expand_dims(image, axis=0)
            
        # 予測
        prediction = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return predicted_class, confidence, prediction[0]
    
    def evaluate(self, x_test, y_test):
        """
        テストデータでモデルを評価
        
        Args:
            x_test: テストデータ
            y_test: テストラベル（one-hot encoded）
            
        Returns:
            評価結果
        """
        print("\n=== モデル評価中 ===")
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"テスト損失: {loss:.4f}")
        print(f"テスト精度: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        
        return loss, accuracy
    
    def plot_predictions(self, x_test, y_test, num_samples=25, save_path='predictions.png'):
        """
        予測結果を可視化
        
        Args:
            x_test: テストデータ
            y_test: テストラベル（one-hot encoded）
            num_samples: 表示するサンプル数
            save_path: 保存先パス
        """
        # ランダムにサンプルを選択
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        
        # 予測
        predictions = self.predict(x_test[indices])
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test[indices], axis=1)
        
        # プロット
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, (idx, ax) in enumerate(zip(indices, axes)):
            # 画像を表示
            ax.imshow(x_test[idx].squeeze(), cmap='gray')
            
            # タイトルを設定（正解/不正解で色分け）
            pred_label = predicted_labels[i]
            true_label = true_labels[i]
            confidence = predictions[i][pred_label]
            
            color = 'green' if pred_label == true_label else 'red'
            ax.set_title(f'予測: {pred_label} ({confidence:.2%})\n正解: {true_label}',
                        color=color, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n予測結果を保存しました: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, x_test, y_test, save_path='confusion_matrix.png'):
        """
        混同行列を作成・可視化
        
        Args:
            x_test: テストデータ
            y_test: テストラベル（one-hot encoded）
            save_path: 保存先パス
        """
        # 予測
        predictions = self.predict(x_test)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        
        # 混同行列の作成
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # 可視化
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('混同行列', fontsize=16, pad=20)
        plt.xlabel('予測ラベル', fontsize=12)
        plt.ylabel('正解ラベル', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n混同行列を保存しました: {save_path}")
        plt.close()
        
        # 分類レポートを出力
        print("\n=== 分類レポート ===")
        report = classification_report(true_labels, predicted_labels, 
                                      target_names=[str(i) for i in range(10)])
        print(report)
        
        return cm
    
    def plot_confidence_distribution(self, x_test, y_test, save_path='confidence_distribution.png'):
        """
        予測信頼度の分布を可視化
        
        Args:
            x_test: テストデータ
            y_test: テストラベル（one-hot encoded）
            save_path: 保存先パス
        """
        # 予測
        predictions = self.predict(x_test)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        
        # 最大確率を取得
        max_confidences = np.max(predictions, axis=1)
        
        # 正解/不正解で分割
        correct_mask = predicted_labels == true_labels
        correct_confidences = max_confidences[correct_mask]
        incorrect_confidences = max_confidences[~correct_mask]
        
        # プロット
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=50, alpha=0.7, color='green', label='正解')
        plt.hist(incorrect_confidences, bins=50, alpha=0.7, color='red', label='不正解')
        plt.xlabel('予測信頼度', fontsize=12)
        plt.ylabel('頻度', fontsize=12)
        plt.title('予測信頼度の分布', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot([correct_confidences, incorrect_confidences], 
                   labels=['正解', '不正解'])
        plt.ylabel('予測信頼度', fontsize=12)
        plt.title('予測信頼度の箱ひげ図', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n信頼度分布を保存しました: {save_path}")
        plt.close()
        
        # 統計情報を表示
        print(f"\n正解時の平均信頼度: {correct_confidences.mean():.4f}")
        print(f"不正解時の平均信頼度: {incorrect_confidences.mean():.4f}")


def load_test_data():
    """テストデータの読み込み"""
    print("=== テストデータ読み込み中 ===")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # 前処理
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"テストデータ: {x_test.shape}")
    return x_test, y_test


def main():
    """メイン処理"""
    # TensorFlowのログレベルを設定
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("=" * 60)
    print("MNIST文字分類 予測・評価")
    print("=" * 60)
    
    # テストデータの読み込み
    x_test, y_test = load_test_data()
    
    # 予測器の初期化
    predictor = MNISTPredictor('models/mnist_model.h5')
    
    # モデルの評価
    predictor.evaluate(x_test, y_test)
    
    # 予測結果の可視化
    print("\n=== 予測結果を可視化中 ===")
    predictor.plot_predictions(x_test, y_test, num_samples=25, 
                              save_path='predictions.png')
    
    # 混同行列の作成
    print("\n=== 混同行列を作成中 ===")
    predictor.plot_confusion_matrix(x_test, y_test, 
                                   save_path='confusion_matrix.png')
    
    # 信頼度分布の可視化
    print("\n=== 信頼度分布を作成中 ===")
    predictor.plot_confidence_distribution(x_test, y_test,
                                          save_path='confidence_distribution.png')
    
    # サンプル予測のデモ
    print("\n=== サンプル予測デモ ===")
    for i in range(5):
        idx = np.random.randint(0, len(x_test))
        image = x_test[idx]
        true_label = np.argmax(y_test[idx])
        
        predicted_class, confidence, probs = predictor.predict_single(image)
        print(f"\nサンプル {i+1}:")
        print(f"  正解ラベル: {true_label}")
        print(f"  予測ラベル: {predicted_class}")
        print(f"  信頼度: {confidence:.4f} ({confidence * 100:.2f}%)")
        print(f"  上位3予測: {np.argsort(probs)[-3:][::-1]}")
    
    print("\n" + "=" * 60)
    print("評価完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
