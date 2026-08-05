import asyncio
import json
import websockets
import os
import numpy as np
from faster_whisper import WhisperModel

HOST = "0.0.0.0"
PORT = 8765

MODEL_PATH = "models/faster-whisper-tiny"
if not os.path.exists(MODEL_PATH):
    print(f"[错误] 找不到本地模型路径: {MODEL_PATH}")
    exit(1)

print(f"[INFO] 正在加载 Whisper 离线模型: {MODEL_PATH}...")
model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
print("[INFO] Whisper 离线引擎加载完成！\n")

async def handle_client(websocket):
    print(f"\n [云端] 客户端已连接: {websocket.remote_address}")
    audio_buffer = bytearray()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                audio_buffer.extend(message)

            elif isinstance(message, str):
                data = json.loads(message)
                if data.get("type") == "END_OF_SPEECH":
                    print(f"\n [云端] 接收到音频 ({len(audio_buffer)} bytes)，开始解析...")
                    
                    if len(audio_buffer) < 3200:
                        audio_buffer = bytearray()
                        continue

                    pcm_data = bytes(audio_buffer)
                    audio_buffer = bytearray()

                    pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
                    audio_float = pcm_array.astype(np.float32) / 32768.0

                    max_amp = np.max(np.abs(pcm_array)) if len(pcm_array) > 0 else 0
                    print(f" [云端] 传入音频安全振幅: {max_amp}")

                    # 开启简体中文强引导
                    segments, info = model.transcribe(
                        audio_float,
                        language="zh",
                        initial_prompt="这是一段普通话语音指令，请输出简体中文：打开灯，关闭灯，今天天气怎么样，今天星期几。",
                        beam_size=5,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        vad_filter=True,
                    )
                    
                    user_text = "".join([segment.text for segment in segments]).strip()
                    print(f" >>> 【识别结果】: '{user_text}' <<<\n")

                    action = "NONE"
                    reply_text = f"听清了，你说的是：{user_text}" if user_text else "抱歉，没听清你说什么。"
                    
                    # 同时匹配简体和繁体指令
                    if ("开" in user_text or "開" in user_text) and ("灯" in user_text or "燈" in user_text):
                        action = "LED_ON"
                        reply_text = "好的，为您打开灯光。"
                    elif ("关" in user_text or "關" in user_text) and ("灯" in user_text or "燈" in user_text):
                        action = "LED_OFF"
                        reply_text = "好的，为您关闭灯光。"

                    response = {
                        "status": "SUCCESS",
                        "user_text": user_text,
                        "reply_text": reply_text,
                        "action": action
                    }
                    await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        print("\n [云端] 客户端断开连接")

async def main():
    server = await websockets.serve(handle_client, HOST, PORT)
    print(f"==================================================")
    print(f"  高灵敏离线 STT 服务已启动: ws://{HOST}:{PORT}")
    print(f"==================================================")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
