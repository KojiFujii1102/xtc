#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude チャットボットのデモンストレーション
実際のAPI呼び出しはせず、動作イメージを表示します
"""

import time

def demo_chatbot():
    print("=" * 50)
    print("Claude チャットボット (コンソール版)")
    print("=" * 50)
    print()
    print("✅ API Keyが設定されています")
    print()
    print("チャットを開始します。終了するには 'exit' または 'quit' と入力してください。")
    print("-" * 50)
    print()
    
    # デモ会話1
    print("あなた: こんにちは！")
    print()
    print("Claude: ", end="", flush=True)
    time.sleep(0.3)
    response = "こんにちは！お手伝いできることがあれば教えてください。どのようなことについてお話ししましょうか？"
    for char in response:
        print(char, end="", flush=True)
        time.sleep(0.02)
    print("\n")
    
    # デモ会話2
    print("あなた: Pythonでファイルを読み込む方法を教えて")
    print()
    print("Claude: ", end="", flush=True)
    time.sleep(0.3)
    response2 = """Pythonでファイルを読み込む基本的な方法をいくつか紹介します：

1. with文を使う方法（推奨）：
```python
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)
```

2. 行ごとに読み込む：
```python
with open('file.txt', 'r', encoding='utf-8') as f:
    for line in f:
        print(line.strip())
```

with文を使うと、自動的にファイルが閉じられるため安全です。"""
    
    for char in response2:
        print(char, end="", flush=True)
        time.sleep(0.01)
    print("\n")
    
    # デモ会話3
    print("あなた: ありがとう！")
    print()
    print("Claude: ", end="", flush=True)
    time.sleep(0.3)
    response3 = "どういたしまして！他に質問があれば、お気軽にどうぞ。"
    for char in response3:
        print(char, end="", flush=True)
        time.sleep(0.02)
    print("\n")
    
    # 終了
    print("あなた: exit")
    print()
    print("チャットを終了します。")
    print()
    print("=" * 50)
    print("デモ終了")
    print("=" * 50)
    print()
    print("📝 実際に使用する場合:")
    print("   1. Anthropic ConsoleでAPI Keyを取得")
    print("   2. export ANTHROPIC_API_KEY='your-key-here'")
    print("   3. python3 claude_chat.py")
    print()
    print("✨ 特徴:")
    print("   - ストリーミングレスポンス（リアルタイム表示）")
    print("   - 会話履歴を保持（文脈を理解）")
    print("   - 日本語・英語対応")
    print("   - エンコーディングエラーなし")

if __name__ == "__main__":
    demo_chatbot()
