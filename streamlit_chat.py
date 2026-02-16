#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import os
from anthropic import Anthropic

# ページ設定
st.set_page_config(
    page_title="Claude チャットボット",
    page_icon="🤖",
    layout="centered"
)

# タイトル
st.title("🤖 Claude チャットボット")

# サイドバーでAPI設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # API Key入力
    api_key = st.text_input(
        "Claude API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic ConsoleでAPI Keyを取得してください"
    )
    
    # モデル選択
    model = st.selectbox(
        "モデル選択",
        [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229"
        ],
        index=0
    )
    
    # Max Tokens
    max_tokens = st.slider("Max Tokens", 256, 4096, 2048, 256)
    
    # チャット履歴をクリア
    if st.button("🗑️ チャット履歴をクリア"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📖 使い方")
    st.markdown("""
    1. API Keyを入力
    2. メッセージを送信
    3. Claudeと会話
    
    **終了**: ページを閉じるかリロード
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 リンク")
    st.markdown("[API Keyを取得](https://console.anthropic.com/)")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力
if prompt := st.chat_input("メッセージを入力してください..."):
    # API Keyのチェック
    if not api_key:
        st.error("⚠️ サイドバーでClaude API Keyを入力してください")
        st.stop()
    
    # ユーザーメッセージを追加
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # アシスタントの応答を生成
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Claudeクライアントの初期化
            client = Anthropic(api_key=api_key)
            
            # ストリーミングレスポンスを取得
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=st.session_state.messages
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    message_placeholder.markdown(full_response + "▌")
            
            # 最終レスポンスを表示
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}"
            st.error(error_msg)
            full_response = error_msg
    
    # アシスタントのメッセージを履歴に追加
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

# フッター
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
    "Powered by Claude (Anthropic)"
    "</div>",
    unsafe_allow_html=True
)
