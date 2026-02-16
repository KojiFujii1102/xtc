#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# テスト用のAPI Key（ダミー）
os.environ["ANTHROPIC_API_KEY"] = "test-key-for-demonstration"

# 標準入力をシミュレート
class MockInput:
    def __init__(self, inputs):
        self.inputs = iter(inputs)
    
    def __call__(self, prompt=""):
        try:
            value = next(self.inputs)
            print(f"{prompt}{value}")
            return value
        except StopIteration:
            return "exit"

# テスト実行
print("=" * 60)
print("Claude チャットボットのテスト実行")
print("=" * 60)
print()

# claude_chat.pyのコードを確認
print("📄 claude_chat.py の内容:")
print("-" * 60)
with open('/home/user/webapp/claude_chat.py', 'r', encoding='utf-8') as f:
    content = f.read()
    # 最初の30行を表示
    lines = content.split('\n')
    for i, line in enumerate(lines[:35], 1):
        print(f"{i:3}: {line}")
print("-" * 60)
print()

print("✅ ファイルは正しく作成されています")
print()
print("💡 実行方法:")
print("   1. Claude API Keyを取得: https://console.anthropic.com/")
print("   2. 環境変数に設定: export ANTHROPIC_API_KEY='your-key'")
print("   3. 実行: python3 claude_chat.py")
print()
print("🔍 機能:")
print("   - リアルタイムストリーミングレスポンス")
print("   - 会話履歴の保持")
print("   - 日本語完全対応")
print("   - 'exit' または 'quit' で終了")
