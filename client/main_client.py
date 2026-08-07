import os
import glob
import asyncio
import websockets
import pyaudio
import numpy as np
import sherpa_onnx
import json

# 获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_BASE = os.path.join(BASE_DIR, "models")

# 自动寻找以 sherpa-onnx-kws 开头的模型目录
matching_dirs = glob.glob(os.path.join(MODELS_BASE, "sherpa-onnx-kws*"))
if matching_dirs and os.path.isdir(matching_dirs[0]):
    MODEL_DIR = matching_dirs[0]
else:
    MODEL_DIR = os.path.join(MODELS_BASE, "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01")

print(f"[配置检查] 成功匹配 KWS 模型路径: {MODEL_DIR}")

# 配置参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1600

SERVER_URI = "ws://localhost:8765"

# 1. 初始化 Sherpa-ONNX 唤醒词检测器
def create_kws_recognizer():
    kws = sherpa_onnx.KeywordSpotter(
        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
        encoder=os.path.join(MODEL_DIR, "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        decoder=os.path.join(MODEL_DIR, "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        joiner=os.path.join(MODEL_DIR, "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        keywords_file=os.path.join(MODEL_DIR, "keywords.txt"),
        num_threads=2,
    )
    return kws

# 2. 音频防过载算法 (保持包络平滑，避免逐帧增益跳变导致的 ASR 识别失真)
def soft_limiter(data_np):
    data_float = data_np.astype(np.float32)
    np.clip(data_float, -30000.0, 30000.0, out=data_float)
    return data_float.astype(np.int16)

async def record_and_send(websocket, stream):
    print("\n>>> 【唤醒成功】 <<<")
    print("[终端] 连接成功！请在 4 秒内清晰说出指令...")
    
    # 清理唤醒词遗留的硬件缓冲音频数据
    try:
        while stream.get_read_available() > CHUNK:
            stream.read(CHUNK, exception_on_overflow=False)
    except Exception:
        pass

    # 录制 4 秒音频并推送
    for _ in range(0, int(RATE / CHUNK * 4)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        data_np = np.frombuffer(data, dtype=np.int16)
        data_processed = soft_limiter(data_np)
        await websocket.send(data_processed.tobytes())
        await asyncio.sleep(0.01)

    print("[终端] 录音完成，正在等待识别与大模型回复...")
    await websocket.send("END_OF_SPEECH")

    response = await websocket.recv()
    res_data = json.loads(response)
    print(f"\n==========================================")
    print(f"收到响应: {res_data.get('reply')}")
    if res_data.get("type") == "action":
        print(f"执行硬件动作: [{res_data.get('action')}]")
    print(f"==========================================\n")

async def main():
    kws = create_kws_recognizer()
    kws_stream = kws.create_stream()

    p = pyaudio.PyAudio()
    mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("==========================================")
    print("系统就绪！对着麦克风喊【小爱同学】")
    print("==========================================")

    while True:
        data = mic_stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        kws_stream.accept_waveform(16000, samples)

        while kws.is_ready(kws_stream):
            kws.decode_stream(kws_stream)

        keyword = kws.get_result(kws_stream)
        if keyword:
            try:
                async with websockets.connect(SERVER_URI) as websocket:
                    await record_and_send(websocket, mic_stream)
            except Exception as e:
                print(f"连接服务端错误: {e}")
            
            kws_stream = kws.create_stream()
            print("系统已重新就绪，随时呼叫【小爱同学】...")

if __name__ == "__main__":
    asyncio.run(main())
