#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
from anthropic import Anthropic
import os

def chat_with_claude(message, history, api_key, model, max_tokens):
    """
    Claudeとチャットする関数
    """
    if not api_key:
        return "⚠️ API Keyを入力してください"
    
    try:
        # Claudeクライアントを初期化
        client = Anthropic(api_key=api_key)
        
        # 会話履歴を構築
        messages = []
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})
        
        # ストリーミングレスポンスを取得
        response_text = ""
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                response_text += text
                yield response_text
        
    except Exception as e:
        yield f"エラーが発生しました: {str(e)}"

# Gradioインターフェースを構築
with gr.Blocks(title="Claude チャットボット", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Claude チャットボット")
    gr.Markdown("Claude APIを使用したチャットボットです。API Keyを入力してチャットを開始してください。")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                label="チャット",
                show_label=True,
                avatar_images=("👤", "🤖")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="メッセージ",
                    placeholder="メッセージを入力してください...",
                    scale=4,
                    lines=2
                )
                send = gr.Button("送信", variant="primary", scale=1)
            
            clear = gr.Button("🗑️ チャット履歴をクリア")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 設定")
            
            api_key = gr.Textbox(
                label="Claude API Key",
                type="password",
                placeholder="sk-ant-...",
                value=os.environ.get("ANTHROPIC_API_KEY", "")
            )
            
            model = gr.Dropdown(
                choices=[
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20240229"
                ],
                value="claude-3-5-sonnet-20241022",
                label="モデル"
            )
            
            max_tokens = gr.Slider(
                minimum=256,
                maximum=4096,
                value=2048,
                step=256,
                label="Max Tokens"
            )
            
            gr.Markdown("---")
            gr.Markdown("### 📖 使い方")
            gr.Markdown("""
            1. API Keyを入力
            2. メッセージを送信
            3. Claudeと会話
            
            [API Keyを取得](https://console.anthropic.com/)
            """)
    
    # イベントハンドラー
    def respond(message, chat_history, api_key, model, max_tokens):
        bot_message = chat_with_claude(message, chat_history, api_key, model, max_tokens)
        chat_history.append((message, ""))
        
        for partial_response in bot_message:
            chat_history[-1] = (message, partial_response)
            yield chat_history
    
    msg.submit(respond, [msg, chatbot, api_key, model, max_tokens], chatbot).then(
        lambda: "", None, msg
    )
    send.click(respond, [msg, chatbot, api_key, model, max_tokens], chatbot).then(
        lambda: "", None, msg
    )
    clear.click(lambda: None, None, chatbot)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
