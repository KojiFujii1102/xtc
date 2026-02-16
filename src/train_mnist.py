#!/usr/bin/env python3
"""
MNIST文字分類 Deep Learning モデル訓練スクリプト

このスクリプトはMNISTデータセットを使用して、
畳み込みニューラルネットワーク（CNN）を訓練します。
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime


class MNISTClassifier:
    """MNIST文字分類のためのCNNモデルクラス"""
    
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """
        初期化
        
        Args:
            input_shape: 入力画像の形状
            num_classes: 分類クラス数（0-9の10クラス）
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """CNNモデルを構築"""
        model = keras.Sequential([
            # 第1畳み込み層
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # 第2畳み込み層
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # 第3畳み込み層
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # 全結合層
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """モデルをコンパイル"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, x_train, y_train, x_val, y_val, 
              epochs=50, batch_size=128, model_path='models/mnist_model.h5'):
        """
        モデルを訓練
        
        Args:
            x_train: 訓練データ
            y_train: 訓練ラベル
            x_val: 検証データ
            y_val: 検証ラベル
            epochs: エポック数
            batch_size: バッチサイズ
            model_path: モデル保存パス
        """
        # コールバックの設定
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        # データ拡張
        data_augmentation = keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1)
        ])
        
        # 訓練
        print("\n=== モデル訓練開始 ===")
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[checkpoint, early_stopping, tensorboard],
            verbose=1
        )
        
        return self.history
    
    def plot_history(self, save_path='training_history.png'):
        """訓練履歴をプロット"""
        if self.history is None:
            print("訓練履歴がありません")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 精度のプロット
        ax1.plot(self.history.history['accuracy'], label='訓練精度')
        ax1.plot(self.history.history['val_accuracy'], label='検証精度')
        ax1.set_title('モデル精度', fontsize=14)
        ax1.set_xlabel('エポック')
        ax1.set_ylabel('精度')
        ax1.legend()
        ax1.grid(True)
        
        # 損失のプロット
        ax2.plot(self.history.history['loss'], label='訓練損失')
        ax2.plot(self.history.history['val_loss'], label='検証損失')
        ax2.set_title('モデル損失', fontsize=14)
        ax2.set_xlabel('エポック')
        ax2.set_ylabel('損失')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n訓練履歴を保存しました: {save_path}")
        plt.close()


def load_and_preprocess_data():
    """MNISTデータセットの読み込みと前処理"""
    print("=== MNISTデータセット読み込み中 ===")
    
    # データセットの読み込み
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"訓練データ: {x_train.shape}")
    print(f"テストデータ: {x_test.shape}")
    
    # データの正規化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # チャンネル次元を追加
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # ラベルをone-hot encodingに変換
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # 訓練データから検証データを分割
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]
    
    print(f"\n前処理後:")
    print(f"訓練データ: {x_train.shape}")
    print(f"検証データ: {x_val.shape}")
    print(f"テストデータ: {x_test.shape}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def main():
    """メイン処理"""
    # TensorFlowのログレベルを設定
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("=" * 60)
    print("MNIST文字分類 Deep Learning モデル訓練")
    print("=" * 60)
    
    # GPUの確認
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU利用可能: {len(gpus)} 台")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("\nCPUで訓練します")
    
    # データの読み込みと前処理
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # モデルの構築
    print("\n=== モデル構築中 ===")
    classifier = MNISTClassifier()
    classifier.build_model()
    classifier.compile_model(learning_rate=0.001)
    
    # モデルサマリーの表示
    print("\n=== モデルアーキテクチャ ===")
    classifier.model.summary()
    
    # モデルの訓練
    history = classifier.train(
        x_train, y_train,
        x_val, y_val,
        epochs=10,
        batch_size=128,
        model_path='models/mnist_model.h5'
    )
    
    # 訓練履歴のプロット
    classifier.plot_history('training_history.png')
    
    # テストデータでの評価
    print("\n=== テストデータで評価中 ===")
    test_loss, test_accuracy = classifier.model.evaluate(x_test, y_test, verbose=0)
    print(f"テスト損失: {test_loss:.4f}")
    print(f"テスト精度: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    # モデルをTensorFlow Lite形式でも保存
    print("\n=== TensorFlow Lite形式で保存中 ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(classifier.model)
    tflite_model = converter.convert()
    
    os.makedirs('models', exist_ok=True)
    with open('models/mnist_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TensorFlow Liteモデルを保存しました: models/mnist_model.tflite")
    
    print("\n" + "=" * 60)
    print("訓練完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
