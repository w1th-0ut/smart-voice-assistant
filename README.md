# Smart Voice Assistant (智能语音助手)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-green.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Ubuntu-orange.svg)
![Status](https://img.shields.io/badge/status-Active%20Development-brightgreen.svg)

基于 Python 打造的高性能轻量级**全离线智能语音助手**系统。结合本地关键词唤醒 (KWS) 与 Faster-Whisper 语音识别 (STT) 引擎，实现毫秒级响应、防过载语音处理以及智能指令控制。

---

## 💡 核心特性

- **100% 离线运行**：无需依赖任何云端 API，保障隐私安全与极低网络延时。
- **高灵敏离线唤醒词 (KWS)**：基于 `sherpa-onnx` 与 Zipformer 轻量模型，支持“小爱同学”唤醒词毫秒级检测。
- **高精度离线语音识别 (STT)**：集成 CTranslate2 优化的 `faster-whisper`，提供高效的中文普通话识别。
- **防过载与音频增强算法**：客户端内置动态音频平滑限幅与软增益控制，彻底解决麦克风爆音/削顶导致的失真与识别幻觉。
- **高并发异步通信架构**：基于 WebSockets 协议构建 Client-Server 端到端实时音频流传输与指令下发通道。
- **智能意图解析与硬件控制**：自动提取语音语义，映射设备控制指令（如 `LED_ON` / `LED_OFF`）。

---

## 🛠️ 架构设计

```text
+-----------------------------------------------------------------------+
|                            Client Side                                |
|  [ Microphone ] ---> [ Audio Soft Limiter ] ---> [ KWS (sherpa-onnx) ]|
|                                                         |             |
|                                                    (Triggered)        |
|                                                         v             |
|                                             [ WebSocket Streamer ]    |
+-----------------------------------------------------------------------+
                                  |
                           (WebSocket / PCM)
                                  v
+-----------------------------------------------------------------------+
|                            Server Side                                |
|  [ WebSocket Receiver ] ---> [ Native float32 Buffer Processing ]     |
|                                         |                             |
|                                         v                             |
|                           [ STT (faster-whisper) ]                    |
|                                         |                             |
|                                         v                             |
|                           [ Intent Extractor ]                        |
|                                         |                             |
|                          [ JSON Action Payload ]                      |
+-----------------------------------------------------------------------+
```

---

## 📂 项目结构

```text
smart-voice-assistant/
├── client/
│   └── main_client.py        # 客户端：音频采集、唤醒词检测、软限幅与 WebSocket 推流
├── server/
│   └── app.py                # 服务端：WebSocket 异步接收、Whisper 离线转写与指令映射
├── models/
│   ├── faster-whisper-tiny/  # Whisper 离线 STT 模型
│   └── sherpa-onnx-kws-.../  # Sherpa-ONNX 离线 KWS 唤醒模型
└── README.md                 # 项目说明文档
```

---

## 🚀 快速开始

### 1. 环境准备

建议使用 Conda 创建独立的 Python 3.10 环境：

```bash
conda create -n voice_env python=3.10 -y
conda activate voice_env
```

安装核心依赖库：

```bash
pip install websockets pyaudio numpy sherpa-onnx faster-whisper
```

*注：Linux 环境下安装 `pyaudio` 前请确保已安装 `portaudio19-dev` (`sudo apt-get install portaudio19-dev`)*。

### 2. 运行服务端

启动离线 STT 语音识别引擎与 WebSocket 通信服务：

```bash
python server/app.py
```

### 3. 运行客户端

启动本地麦克风监听与唤醒模块：

```bash
python client/main_client.py
```

### 4. 交互体验

1. 终端提示 `系统就绪！对着麦克风喊【小爱同学】`。
2. 清晰喊出 **“小爱同学”**，看到 `>>> 【唤醒成功】 <<<` 提示。
3. 在 4 秒录音时间内说出指令，例如：
   - **“打开灯”** $
ightarrow$ 返回动作代码 `LED_ON`
   - **“关闭灯光”** $
ightarrow$ 返回动作代码 `LED_OFF`
   - **“今天天气怎么样”** $
ightarrow$ 返回意图识别文本

---

## 📈 项目进度看板

- [x] **2026-08-04**: 完成 PyAudio 原始音频采集与音频硬件输入调试。
- [x] **2026-08-04**: 完成客户端动态音频平滑限幅算法，修复波形削顶过载爆音问题。
- [x] **2026-08-04**: 集成 `sherpa-onnx` 离线唤醒词引擎，实现“小爱同学”稳定触发。
- [x] **2026-08-04**: 搭建 WebSockets 全双工异步推流通信架构。
- [x] **2026-08-04**: 集成 `faster-whisper` 离线识别，解决内存泄漏/幻觉/繁体转码问题。
- [x] **2026-08-04**: 完成首个阶段意图匹配与硬件控制指令 Payload 输出。
- [ ] **NEXT**: 对接本地开源 LLM（如 Qwen-2.5 / Ollama）实现自由问答与复杂逻辑推理。
- [ ] **NEXT**: 集成轻量级离线 TTS 实现语音播报。
- [ ] **NEXT**: 对接 ESP32 / 树莓派 GPIO 实现实体硬件联动。

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源协议。
