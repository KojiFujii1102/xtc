#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from anthropic import Anthropic

def main():
    print("=" * 50)
    print("Claude チャットボット (コンソール版)")
    print("=" * 50)
    print()
    
    # API Keyの取得
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Claude API Keyを入力してください: ").strip()
    
    if not api_key:
        print("エラー: API Keyが必要です")
        return
    
    # Claudeクライアントの初期化
    client = Anthropic(api_key=api_key)
    
    # チャット履歴
    conversation_history = []
    
    print("\nチャットを開始します。終了するには 'exit' または 'quit' と入力してください。")
    print("-" * 50)
    print()
    
    while True:
        # ユーザー入力
        user_input = input("あなた: ")
        
        # 終了チェック
        if user_input.lower() in ['exit', 'quit', 'q', '終了']:
            print("\nチャットを終了します。")
            break
        
        if not user_input.strip():
            continue
        
        # 履歴に追加
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        try:
            # Claude APIを呼び出し
            print("\nClaude: ", end="", flush=True)
            
            response_text = ""
            with client.messages.stream(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=conversation_history
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    response_text += text
            
            print("\n")
            
            # 履歴に追加
            conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            # エラーが発生した場合、最後のユーザーメッセージを削除
            conversation_history.pop()
            print()

if __name__ == "__main__":
    main()
