import asyncio
import json
import websockets
import numpy as np
from faster_whisper import WhisperModel
import ollama

# 1. 加载 Faster-Whisper 离线模型 (优先采用 small 模型提高中文识别准确度)
import os
model_path = "models/faster-whisper-small" if os.path.exists("models/faster-whisper-small") else "models/faster-whisper-tiny"
print(f"正在加载 Faster-Whisper 离线模型 ({model_path})...")
stt_model = WhisperModel(model_path, device="cpu", compute_type="int8")
print("Whisper 模型加载完成！")

# 2. 带有【实时时间上下文 + 语音自动纠错与直接回答】的大模型 Prompt
import datetime

def get_system_prompt():
    now = datetime.datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday_str = weekdays[now.weekday()]
    time_str = now.strftime("%Y年%m月%d日 %H:%M")
    
    return (
        "你是一个贴心的智能语音助手AI。\n"
        f"【当前系统真实时间】：{time_str} {weekday_str}\n"
        "【任务要求】：\n"
        "1. 用户的输入来自语音识别，可能存在同音字或错别字，请自动纠正并理解真实意图。\n"
        "2. 给出具体的回答或解决方案，绝不能反问或原样重复用户的提问！如果询问时间或星期，请结合【当前系统真实时间】准确回答。\n"
        "3. 用极简短、自然、温和的口语回答（25字以内）。\n"
        "4. 严禁 Markdown 格式，严禁解释纠错过程。"
    )

def ask_llm(prompt):
    try:
        response = ollama.chat(
            model='qwen2.5:1.5b',
            messages=[
                {'role': 'system', 'content': get_system_prompt()},
                {'role': 'user', 'content': prompt}
            ],
            options={
                'num_predict': 50,
                'temperature': 0.3
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"大模型响应失败: {str(e)}"

# 3. 意图解析与回复策略
def process_intent(text):
    text_clean = text.lower().strip()
    
    # 硬件控制指令保留强匹配规则
    if ("打" in text_clean or "开" in text_clean) and "灯" in text_clean:
        return {"type": "action", "action": "LED_ON", "reply": "好的，已为您打开灯光。"}
    elif ("关" in text_clean or "闭" in text_clean) and "灯" in text_clean:
        return {"type": "action", "action": "LED_OFF", "reply": "好的，已为您关闭灯光。"}
    
    # 其他所有语句（包括模糊/识别有误的句子）全部交给智能大模型进行自动纠错与回答
    print(f"--> [原始语音识别结果]: '{text}'")
    ai_reply = ask_llm(text)
    return {"type": "chat", "reply": ai_reply}

# 4. WebSocket 服务端处理逻辑
async def audio_handler(websocket):
    print("\n[Client Connected] 客户端已连接 WebSocket！")
    audio_buffer = []

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(chunk)
            elif message == "END_OF_SPEECH":
                print("\n[Server] 接收到音频结束信号，开始转写...")
                if not audio_buffer:
                    continue
                
                full_audio = np.concatenate(audio_buffer)
                audio_buffer = [] # 清空缓存

                # 执行 100% 离线语音转写 (设定 initial_prompt 提示 Whisper 优先输出简体中文)
                segments, _ = stt_model.transcribe(
                    full_audio,
                    beam_size=5,
                    language="zh",
                    initial_prompt="这是一段普通话语音对话，请使用简体中文输出。",
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                recognized_text = "".join([segment.text for segment in segments]).strip()
                print(f"[识别结果]: '{recognized_text}'")

                if recognized_text:
                    result = process_intent(recognized_text)
                    print(f"[AI 智能回复]: {result['reply']}")
                    await websocket.send(json.dumps(result, ensure_ascii=False))
                else:
                    await websocket.send(json.dumps({"type": "error", "reply": "未能识别到声音"}))

    except websockets.exceptions.ConnectionClosed:
        print("[Client Disconnected] 客户端断开连接")

async def main():
    server = await websockets.serve(audio_handler, "0.0.0.0", 8765)
    print("==========================================")
    print("语音服务端启动成功！监听端口: 8765")
    print("已启用: 简体中文强制引导 + LLM 语音同音字自动纠错引擎")
    print("==========================================")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
