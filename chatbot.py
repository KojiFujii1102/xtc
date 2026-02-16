import streamlit as st
import openai
import os
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="シンプルチャットボット",
    page_icon="🤖",
    layout="centered"
)

# タイトル
st.title("🤖 シンプルLLMチャットボット")

# サイドバーでAPI設定
with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    
    model = st.selectbox(
        "モデル選択",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        index=0
    )
    
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    
    if st.button("チャット履歴をクリア"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 使い方")
    st.markdown("""
    1. OpenAI API Keyを入力
    2. メッセージを入力して送信
    3. AIからの返答を確認
    """)

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
        st.error("⚠️ OpenAI API Keyを入力してください")
        st.stop()
    
    # ユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # アシスタントの応答を生成
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # OpenAI APIクライアントの設定
            client = openai.OpenAI(api_key=api_key)
            
            # ストリーミングレスポンスを取得
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                temperature=temperature,
                stream=True,
            )
            
            # レスポンスをストリーミング表示
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            full_response = f"エラー: {str(e)}"
    
    # アシスタントのメッセージを履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# フッター
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Powered by OpenAI GPT</div>",
    unsafe_allow_html=True
)
