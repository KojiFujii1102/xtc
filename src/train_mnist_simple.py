#!/usr/bin/env python3
"""
MNIST文字分類 軽量Deep Learning モデル訓練スクリプト

シンプルで軽量なCNNモデルを使用します。
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')  # GUIなしのバックエンドを使用
import matplotlib.pyplot as plt


def build_simple_model(input_shape=(28, 28, 1), num_classes=10):
    """シンプルなCNNモデルを構築"""
    model = keras.Sequential([
        # 第1畳み込み層
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # 第2畳み込み層
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # 全結合層
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


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
    
    print(f"\n前処理後:")
    print(f"訓練データ: {x_train.shape}")
    print(f"テストデータ: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def plot_history(history, save_path='training_history.png'):
    """訓練履歴をプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 精度のプロット
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # 損失のプロット
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n訓練履歴を保存しました: {save_path}")
    plt.close()


def main():
    """メイン処理"""
    # TensorFlowのログレベルを設定
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("=" * 60)
    print("MNIST文字分類 軽量Deep Learning モデル訓練")
    print("=" * 60)
    
    # GPUの確認
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU利用可能: {len(gpus)} 台")
    else:
        print("\nCPUで訓練します")
    
    # データの読み込みと前処理
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # モデルの構築
    print("\n=== モデル構築中 ===")
    model = build_simple_model()
    
    # モデルサマリーの表示
    print("\n=== モデルアーキテクチャ ===")
    model.summary()
    
    # モデルの訓練
    print("\n=== モデル訓練開始 ===")
    
    os.makedirs('models', exist_ok=True)
    
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=5,
        verbose=1,
        validation_split=0.1
    )
    
    # モデルの保存
    model_path = 'models/mnist_model.h5'
    model.save(model_path)
    print(f"\nモデルを保存しました: {model_path}")
    
    # 訓練履歴のプロット
    plot_history(history, 'training_history.png')
    
    # テストデータでの評価
    print("\n=== テストデータで評価中 ===")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"テスト損失: {test_loss:.4f}")
    print(f"テスト精度: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    # TensorFlow Lite形式でも保存
    print("\n=== TensorFlow Lite形式で保存中 ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = 'models/mnist_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TensorFlow Liteモデルを保存しました: {tflite_path}")
    
    print("\n" + "=" * 60)
    print("訓練完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
