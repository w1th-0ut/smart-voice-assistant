import sys
import os
import json
import asyncio
import websockets
import pyaudio
import numpy as np
import sherpa_onnx

SERVER_URI = "ws://127.0.0.1:8765"

async def record_and_send(stream, sample_rate, chunk_size, record_seconds=4):
    try:
        async with websockets.connect(SERVER_URI) as websocket:
            print(f" [终端] 连接成功！请在 4 秒内清晰说出指令...")
            chunks_to_record = int(sample_rate / chunk_size * record_seconds)
            
            pcm_frames = []
            for _ in range(chunks_to_record):
                data = stream.read(chunk_size, exception_on_overflow=False)
                pcm_frames.append(data)
                await asyncio.sleep(0.001)

            # 合并完整录音并做防过载平滑处理
            raw_pcm = b''.join(pcm_frames)
            audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
            
            # 计算音频峰值，如果爆音则自动按比例缩小
            max_val = np.max(np.abs(audio_np))
            if max_val > 10000:
                audio_np = audio_np * (12000.0 / max_val) # 强行降音量防破音
                
            safe_pcm = audio_np.astype(np.int16).tobytes()

            print(" [终端] 录音完成，正在等待识别...")
            await websocket.send(safe_pcm)
            await websocket.send(json.dumps({"type": "END_OF_SPEECH"}))

            response_raw = await websocket.recv()
            response = json.loads(response_raw)
            print("\n==============================================")
            print(f" [终端] 识别结果 : '{response.get('user_text')}'")
            print(f" [终端] AI 响应   : {response.get('reply_text')}")
            print(f" [终端] 硬件指令 : {response.get('action')}")
            print("==============================================\n")

    except Exception as e:
        print(f" [错误] 推流异常: {e}")

def main():
    model_dir = "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    
    kws = sherpa_onnx.KeywordSpotter(
        tokens=os.path.join(model_dir, "tokens.txt"),
        encoder=os.path.join(model_dir, "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        decoder=os.path.join(model_dir, "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        joiner=os.path.join(model_dir, "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        keywords_file=os.path.join(model_dir, "keywords.txt"),
        num_threads=2,
        provider="cpu",
    )

    sample_rate = 16000
    chunk_size = 1600
    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )

    stream_kws = kws.create_stream()

    print("\n==============================================")
    print("  系统就绪！对着麦克风喊【小爱同学】")
    print("==============================================\n")

    try:
        while True:
            samples_bytes = stream.read(chunk_size, exception_on_overflow=False)
            
            samples_int16 = np.frombuffer(samples_bytes, dtype=np.int16)
            samples_float = samples_int16.astype(np.float32) / 32768.0

            stream_kws.accept_waveform(sample_rate, samples_float)

            while kws.is_ready(stream_kws):
                kws.decode_stream(stream_kws)

            result = kws.get_result(stream_kws)
            if result:
                print(f"\n >>> 【唤醒成功】 <<<")
                asyncio.run(record_and_send(stream, sample_rate, chunk_size, record_seconds=4))
                stream_kws = kws.create_stream()

    except KeyboardInterrupt:
        print("\n[INFO] 已安全退出")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
