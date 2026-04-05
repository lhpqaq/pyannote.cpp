# pyannote.cpp

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

High-performance C/C++ inference engine for [pyannote-audio](https://github.com/pyannote/pyannote-audio) speaker diarization, built on ggml.

### What is pyannote.cpp?

pyannote.cpp is a pure C++ implementation of the pyannote-audio speaker diarization pipeline, optimized for CPU inference without requiring Python at runtime. It answers the question "who spoke when?" by:

1. **Voice Activity Detection (VAD)**: Detecting speech segments using PyanNet (SincNet + LSTM)
2. **Speaker Embedding**: Extracting speaker features using Mel-filterbanks + ResNet34
3. **Clustering**: Assigning speaker IDs via greedy cosine-distance clustering

### Features

- **Pure C++ Inference**: No Python dependency at runtime
- **GGML-based**: Leverages the efficient ggml tensor library
- **GGUF Format**: Models stored in the portable GGUF format
- **Cross-platform**: Supports x86_64, ARM, Apple Silicon
- **CPU Optimized**: Efficient CPU inference (GPU support planned)

### Prerequisites

#### For Model Conversion (Python required)
```bash
pip install torch numpy gguf asteroid-filterbanks
```

#### For Building (C++ toolchain)
- CMake 3.10+
- GCC/Clang or MSVC
- Git

### Quick Start

#### 1. Download and Convert Models

Download the PyTorch models from HuggingFace:
- Segmentation model: `pyannote/speaker-diarization-3.1` (subfolder `segmentation/`)
- Embedding model: `pyannote/speaker-diarization-3.1` (subfolder `embedding/`)

Or use the automated download script:
```bash
python download_models.py
```

Convert to GGUF format:
```bash
# Convert segmentation model
python examples/python/convert_pyannote_to_ggml.py \
  models/segmentation/pytorch_model.bin \
  models/pyannote-segmentation.gguf

# Convert embedding model
python examples/python/convert_embedding_to_ggml.py \
  models/embedding/pytorch_model.bin \
  models/pyannote-embedding.gguf

# Convert PLDA model
python examples/python/convert_plda_to_gguf.py \
  --transform-npz models/plda/xvec_transform.npz \
  --plda-npz models/plda/plda.npz \
  -o models/plda.gguf
```

#### 2. Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

On macOS:
```bash
cmake ..
make -j$(sysctl -n hw.ncpu)
```

**Note**: The offline CLI is the supported entry point. Backend availability depends on your local `ggml` build configuration and platform.

#### 3. Run Inference

```bash
./build/bin/pyannote-diarization \
  models/pyannote-segmentation.gguf \
  models/pyannote-embedding.gguf \
  audio.wav \
  --plda models/plda.gguf \
  --output output.rttm
```

**Note**: Input audio must be 16kHz WAV format.

### Output Format

```
[00:00.500 --> 00:04.250] SPEAKER_00
[00:04.500 --> 00:08.750] SPEAKER_01
[00:09.000 --> 00:12.500] SPEAKER_00
```

### Project Structure

```
pyannote.cpp/
├── src/               # ggml core library
├── include/           # ggml headers
├── examples/          # pyannote implementation
│   ├── main.cpp       # main diarization pipeline
│   ├── vbx.cpp        # clustering algorithm
│   ├── python/        # model conversion scripts
│   └── CMakeLists.txt
├── models/            # converted GGUF models (you create this)
└── CMakeLists.txt
```

### License

This project is based on [ggml](https://github.com/ggerganov/ggml) and follows the MIT License. See [LICENSE](LICENSE) for details.

### Acknowledgments

- [pyannote-audio](https://github.com/pyannote/pyannote-audio): Original PyTorch implementation
- [ggml](https://github.com/ggerganov/ggml): Efficient tensor library for machine learning

### Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

<a name="chinese"></a>
## 中文

基于 ggml 实现的高性能 C/C++ [pyannote-audio](https://github.com/pyannote/pyannote-audio) 说话人日志推理引擎。

### 什么是 pyannote.cpp？

pyannote.cpp 是 pyannote-audio 说话人日志流水线的纯 C++ 实现，针对 CPU 推理进行了优化，运行时无需 Python 依赖。它通过以下步骤回答"谁在何时说话"的问题：

1. **语音活动检测 (VAD)**：使用 PyanNet (SincNet + LSTM) 检测语音片段
2. **说话人特征提取**：使用 Mel 滤波器组 + ResNet34 提取说话人特征
3. **聚类**：通过贪心余弦距离聚类分配说话人 ID

### 特性

- **纯 C++ 推理**：运行时无需 Python 依赖
- **基于 GGML**：利用高效的 ggml 张量库
- **GGUF 格式**：模型以可移植的 GGUF 格式存储
- **跨平台**：支持 x86_64、ARM、Apple Silicon
- **CPU 优化**：高效的 CPU 推理（GPU 支持规划中）

### 环境要求

#### 模型转换（需要 Python）
```bash
pip install torch numpy gguf asteroid-filterbanks
```

#### 编译构建（需要 C++ 工具链）
- CMake 3.10+
- GCC/Clang 或 MSVC
- Git

### 快速开始

#### 1. 下载和转换模型

从 HuggingFace 下载 PyTorch 模型：
- 分割模型：`pyannote/speaker-diarization-3.1`（`segmentation/` 子目录）
- 嵌入模型：`pyannote/speaker-diarization-3.1`（`embedding/` 子目录）

或使用自动下载脚本：
```bash
python download_models.py
```

转换为 GGUF 格式：
```bash
# 转换分割模型
python examples/python/convert_pyannote_to_ggml.py \
  models/segmentation/pytorch_model.bin \
  models/pyannote-segmentation.gguf

# 转换嵌入模型
python examples/python/convert_embedding_to_ggml.py \
  models/embedding/pytorch_model.bin \
  models/pyannote-embedding.gguf

python examples/python/convert_plda_to_gguf.py \
  --transform-npz models/plda/xvec_transform.npz \
  --plda-npz models/plda/plda.npz \
  -o models/plda.gguf
```

#### 2. 编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

macOS 系统：
```bash
cmake ..
make -j$(sysctl -n hw.ncpu)
```

**注意**：当前对外支持的入口是离线 CLI。具体 backend 是否可用，取决于你本地的 `ggml` 编译配置和平台。

#### 3. 运行推理

```bash
./build/bin/pyannote-diarization \
  models/pyannote-segmentation.gguf \
  models/pyannote-embedding.gguf \
  audio.wav \
  --plda models/plda.gguf \
  --output output.rttm
```

**注意**：输入音频必须是 16kHz WAV 格式。

### 输出格式

```
[00:00.500 --> 00:04.250] SPEAKER_00
[00:04.500 --> 00:08.750] SPEAKER_01
[00:09.000 --> 00:12.500] SPEAKER_00
```

### 项目结构

```
pyannote.cpp/
├── src/               # ggml 核心库
├── include/           # ggml 头文件
├── examples/          # pyannote 实现
│   ├── main.cpp       # 主要日志流水线
│   ├── vbx.cpp        # 聚类算法
│   ├── python/        # 模型转换脚本
│   └── CMakeLists.txt
├── models/            # 转换后的 GGUF 模型（需自行创建）
└── CMakeLists.txt
```

### 许可证

本项目基于 [ggml](https://github.com/ggerganov/ggml)，遵循 MIT 许可证。详见 [LICENSE](LICENSE)。

### 致谢

- [pyannote-audio](https://github.com/pyannote/pyannote-audio)：原始 PyTorch 实现
- [ggml](https://github.com/ggerganov/ggml)：高效的机器学习张量库

### 贡献

欢迎贡献！请随时提交 issue 和 pull request。
