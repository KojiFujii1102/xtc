#!/usr/bin/env python3
"""
MNIST文字分類 Webインターフェース

Streamlitを使用したMNIST文字分類のWebアプリケーション
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


@st.cache_resource
def load_model(model_path='models/mnist_model.h5'):
    """モデルをキャッシュ付きで読み込み"""
    if not os.path.exists(model_path):
        return None
    return keras.models.load_model(model_path)


def preprocess_image(image):
    """
    アップロードされた画像を前処理
    
    Args:
        image: PIL Image
        
    Returns:
        前処理済みの画像データ
    """
    # グレースケールに変換
    image = ImageOps.grayscale(image)
    
    # 28x28にリサイズ
    image = image.resize((28, 28))
    
    # NumPy配列に変換
    img_array = np.array(image)
    
    # 白黒を反転（MNISTは黒背景に白文字）
    img_array = 255 - img_array
    
    # 正規化
    img_array = img_array.astype('float32') / 255.0
    
    # 形状を調整 (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array


def plot_prediction_chart(probabilities):
    """予測確率をグラフ化"""
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(range(10))
    colors = ['green' if p == max(probabilities) else 'lightblue' for p in probabilities]
    
    ax.bar(classes, probabilities, color=colors)
    ax.set_xlabel('数字', fontsize=12)
    ax.set_ylabel('予測確率', fontsize=12)
    ax.set_title('各クラスの予測確率', fontsize=14)
    ax.set_xticks(classes)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 最大値にラベルを追加
    max_idx = np.argmax(probabilities)
    ax.text(max_idx, probabilities[max_idx] + 0.02, 
           f'{probabilities[max_idx]:.2%}', 
           ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    return fig


def main():
    """メインアプリケーション"""
    # ページ設定
    st.set_page_config(
        page_title="MNIST文字分類",
        page_icon="🔢",
        layout="wide"
    )
    
    # タイトル
    st.title("🔢 MNIST手書き数字分類")
    st.markdown("---")
    
    # サイドバー
    st.sidebar.header("⚙️ 設定")
    
    # モデルの読み込み
    model_path = 'models/mnist_model.h5'
    
    with st.spinner("モデル読み込み中..."):
        model = load_model(model_path)
    
    if model is None:
        st.error(f"❌ モデルが見つかりません: {model_path}")
        st.info("先に `python src/train_mnist.py` でモデルを訓練してください。")
        return
    
    st.sidebar.success("✅ モデル読み込み完了")
    
    # 使い方の説明
    st.sidebar.markdown("""
    ### 📝 使い方
    1. 手書き数字の画像をアップロード
    2. モデルが自動で予測を実行
    3. 結果を確認
    
    ### 📋 対応形式
    - PNG, JPG, JPEG
    - グレースケールまたはカラー
    - 任意のサイズ（自動で28x28にリサイズ）
    """)
    
    # メインエリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 画像アップロード")
        
        # ファイルアップロード
        uploaded_file = st.file_uploader(
            "手書き数字の画像を選択してください",
            type=['png', 'jpg', 'jpeg'],
            help="画像は自動で28x28ピクセルにリサイズされます"
        )
        
        # MNISTサンプルデータの使用オプション
        use_sample = st.checkbox("MNISTサンプルデータを使用")
        
        if use_sample:
            # MNISTからランダムにサンプルを取得
            (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
            sample_idx = st.slider("サンプル番号", 0, len(x_test)-1, 0)
            
            sample_image = x_test[sample_idx]
            sample_label = y_test[sample_idx]
            
            st.image(sample_image, caption=f"サンプル画像 (正解: {sample_label})", 
                    width=200, clamp=True)
            
            # 前処理
            img_array = sample_image.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            img_array = np.expand_dims(img_array, axis=-1)
            
        elif uploaded_file is not None:
            # アップロードされた画像を表示
            image = Image.open(uploaded_file)
            st.image(image, caption="アップロードされた画像", width=200)
            
            # 前処理
            img_array = preprocess_image(image)
            
            # 前処理後の画像を表示
            st.image(img_array[0].squeeze(), caption="前処理後（28x28）", 
                    width=200, clamp=True)
        else:
            img_array = None
    
    with col2:
        st.header("🎯 予測結果")
        
        if img_array is not None:
            # 予測実行
            with st.spinner("予測中..."):
                predictions = model.predict(img_array, verbose=0)[0]
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]
            
            # 結果を表示
            st.markdown(f"""
            ### 予測された数字: **{predicted_class}**
            ### 信頼度: **{confidence:.2%}**
            """)
            
            # 信頼度メーター
            st.progress(float(confidence))
            
            # 信頼度による判定
            if confidence > 0.9:
                st.success("✅ 非常に高い信頼度で予測しました")
            elif confidence > 0.7:
                st.info("ℹ️ 高い信頼度で予測しました")
            elif confidence > 0.5:
                st.warning("⚠️ 中程度の信頼度です")
            else:
                st.error("❌ 信頼度が低いです。画像を確認してください")
            
            # 上位3予測
            st.markdown("---")
            st.subheader("📊 上位3予測")
            top3_indices = np.argsort(predictions)[-3:][::-1]
            
            for i, idx in enumerate(top3_indices):
                st.write(f"{i+1}. 数字 **{idx}**: {predictions[idx]:.2%}")
            
            # 全クラスの予測確率グラフ
            st.markdown("---")
            st.subheader("📈 全クラスの予測確率")
            fig = plot_prediction_chart(predictions)
            st.pyplot(fig)
            
        else:
            st.info("👈 左側から画像をアップロードしてください")
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>MNIST手書き数字分類 Deep Learning モデル</p>
        <p>TensorFlow/Keras + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
