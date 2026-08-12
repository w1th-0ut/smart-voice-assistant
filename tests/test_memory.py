import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 模拟 faster_whisper 避免测试时加载大模型权重
sys.modules['faster_whisper'] = MagicMock()

from server.app import get_system_prompt, ask_llm, process_intent

class TestSlidingWindowMemory(unittest.TestCase):

    @patch('server.app.ollama.chat')
    def test_ask_llm_message_structure(self, mock_ollama_chat):
        mock_ollama_chat.return_value = {'message': {'content': '我是智能助手。'}}
        
        history = [
            {'role': 'user', 'content': '你好'},
            {'role': 'assistant', 'content': '你好！有什么我可以帮你的？'}
        ]
        prompt = "我刚才说了什么？"
        
        reply = ask_llm(prompt, history)
        
        self.assertEqual(reply, '我是智能助手。')
        mock_ollama_chat.assert_called_once()
        _, kwargs = mock_ollama_chat.call_args
        messages = kwargs['messages']
        
        # 1. 第一条必须是 system 消息
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn("你是一个贴心的智能语音助手AI", messages[0]['content'])
        
        # 2. 中间包含传入的历史
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])
        
        # 3. 最后一条是当前 user 消息
        self.assertEqual(messages[3], {'role': 'user', 'content': prompt})

    def test_process_intent_reset(self):
        res1 = process_intent("清空历史")
        self.assertEqual(res1["type"], "reset")
        self.assertEqual(res1["reply"], "已为您清空对话历史。")

        res2 = process_intent("重新开始对话")
        self.assertEqual(res2["type"], "reset")

    def test_process_intent_action(self):
        res = process_intent("请帮我打开客厅的灯")
        self.assertEqual(res["type"], "action")
        self.assertEqual(res["action"], "LED_ON")

    def test_sliding_window_truncation(self):
        # 模拟 WebSocket 会话中的滑动窗口截断机制（最大 6 条 / 3 轮）
        session_history = []
        
        for i in range(5): # 模拟 5 轮对话
            user_msg = f"问题 {i+1}"
            ai_msg = f"回答 {i+1}"
            
            session_history.append({'role': 'user', 'content': user_msg})
            session_history.append({'role': 'assistant', 'content': ai_msg})
            
            if len(session_history) > 6:
                session_history = session_history[-6:]
                
        # 结果应该只包含后 3 轮（第 3、4、5 轮，共 6 条消息）
        self.assertEqual(len(session_history), 6)
        self.assertEqual(session_history[0]['content'], "问题 3")
        self.assertEqual(session_history[-1]['content'], "回答 5")

if __name__ == "__main__":
    unittest.main()
