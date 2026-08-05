import sys
import os
import pyaudio
import numpy as np
import sherpa_onnx

def main():
    model_dir = "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    encoder = os.path.join(model_dir, "encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    decoder = os.path.join(model_dir, "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    joiner = os.path.join(model_dir, "joiner-epoch-12-avg-2-chunk-16-left-64.onnx")
    keywords_file = os.path.join(model_dir, "keywords.txt")

    if not os.path.exists(encoder):
        print(f"[错误] 找不到模型文件，请检查 {model_dir} 路径！")
        return

    print("[INFO] 正在初始化 Sherpa-onnx 离线唤醒引擎...")
    kws = sherpa_onnx.KeywordSpotter(
        tokens=os.path.join(model_dir, "tokens.txt"),
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        keywords_file=keywords_file,
        num_threads=2,
        provider="cpu",
    )

    sample_rate = 16000
    chunk_size = 1600  # 100ms 帧大小
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
    print("  系统就绪！请对着麦克风清晰喊一声: 【小爱同学】")
    print("  按 Ctrl+C 退出测试")
    print("==============================================\n")

    try:
        while True:
            # 1. 采集 PCM 数据
            samples = stream.read(chunk_size, exception_on_overflow=False)
            samples_float = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 2. 喂入音频帧
            stream_kws.accept_waveform(sample_rate, samples_float)

            # 3. 驱动解码器解码
            while kws.is_ready(stream_kws):
                kws.decode_stream(stream_kws)

            # 4. 获取唤醒结果
            result = kws.get_result(stream_kws)
            if result:
                print(f"\n >>> 【唤醒成功！】检测到关键字: {result} <<< \n")
                # 触发唤醒后重置流，准备下一次检测
                kws.reset(stream_kws)

    except KeyboardInterrupt:
        print("\n[INFO] 已退出唤醒测试。")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
