#!/usr/bin/env python3
"""
MNIST文字分類 超軽量Deep Learning モデル訓練スクリプト

最小限のメモリで動作するモデルです。
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_tiny_model(input_shape=(28, 28, 1), num_classes=10):
    """超軽量CNNモデル"""
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    print("=" * 60)
    print("MNIST Deep Learning Model Training (Memory Optimized)")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Use smaller dataset to reduce memory
    x_train = x_train[:10000]  # Use only 10k samples
    y_train = y_train[:10000]
    x_test = x_test[:2000]      # Use only 2k test samples
    y_test = y_test[:2000]
    
    # Preprocess
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_tiny_model()
    model.summary()
    
    # Train
    print("\nTraining...")
    os.makedirs('models', exist_ok=True)
    
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=3,
        verbose=1,
        validation_split=0.1
    )
    
    # Save
    model_path = 'models/mnist_model.h5'
    model.save(model_path)
    print(f"\nModel saved: {model_path}")
    
    # Evaluate
    print("\nEvaluating on test data...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    print("Training history saved: training_history.png")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
