# Pyannote Audio Speaker Diarization in GGML

This project implements a complete, high-performance C++ inference pipeline for [pyannote-audio](https://github.com/pyannote/pyannote-audio) speaker diarization using `ggml`. It includes Voice Activity Detection (Segmentation), Speaker Embedding extraction (ResNet), and Clustering.

## Features
- **Pure C++ Inference**: No Python dependency during runtime.
- **Full Pipeline**:
  1. **Segmentation**: PyanNet (SincNet + LSTM) for detecting active speech segments.
  2. **Feature Extraction**: Mel-filterbanks + ResNet34 for extracting speaker embeddings.
  3. **Clustering**: Greedy cosine-distance clustering to assign global speaker IDs.
- **GGUF Format**: Uses the GGUF file format for efficient model loading.

## Prerequisites

### 1. Python Environment (For Model Conversion only)
You need Python to convert the PyTorch weights to GGUF format.

```bash
pip install torch numpy gguf asteroid-filterbanks
```

### 2. C++ Build Environment
- CMake
- GCC/Clang
- `ggml` dependencies (pthreads, etc.)

---

## Step 1: Prepare Models

You need to download the pre-trained models from HuggingFace and convert them.

**1. Download PyTorch Checkpoints**
Download `pytorch_model.bin` for both segmentation and embedding models.
- Segmentation: `pyannote/speaker-diarization-community-1` (subfolder `segmentation/`)
- Embedding: `pyannote/speaker-diarization-community-1` (subfolder `embedding/`)

**2. Convert to GGUF**

Use the provided scripts in `examples/pyannote/python/`:

```bash
# 1. Convert Segmentation Model
python examples/pyannote/python/convert_pyannote_to_ggml.py \
  /path/to/segmentation/pytorch_model.bin \
  examples/pyannote/pyannote-segmentation.gguf

# 2. Convert Embedding Model
python examples/pyannote/python/convert_embedding_to_ggml.py \
  /path/to/embedding/pytorch_model.bin \
  examples/pyannote/pyannote-embedding.gguf
```

---

## Step 2: Build

Compile the project using CMake.

```bash
mkdir build
cd build
cmake ..
make -j4 pyannote-diarization
```

---

## Step 3: Run Inference

Run the generated binary. The input audio **must be 16kHz WAV format**.

**Usage:**
```bash
./bin/pyannote-diarization <segmentation_model.gguf> <embedding_model.gguf> <input_audio.wav>
```

**Example:**
```bash
./bin/pyannote-diarization \
  ../examples/pyannote/pyannote-segmentation.gguf \
  ../examples/pyannote/pyannote-embedding.gguf \
  ../samples/jfk.wav
```

---

# Pyannote 说话人日志 (中文说明)

本项目基于 `ggml` 实现了 [pyannote-audio](https://github.com/pyannote/pyannote-audio) 的完整 C++ 推理流程。它包含语音活动检测（分割）、声纹特征提取（Embedding）以及聚类，能够在 CPU 上高效运行。

## 主要功能
- **纯 C++ 推理**：运行时无需 Python 环境。
- **完整流水线**：
  1. **分割 (Segmentation)**：使用 PyanNet (SincNet + LSTM) 检测语音片段。
  2. **特征提取 (Embedding)**：使用 Mel 滤波器组 + ResNet34 提取声纹特征。
  3. **聚类 (Clustering)**：基于余弦距离的贪婪聚类算法，生成全局说话人 ID。
- **GGUF 格式**：使用 GGUF 格式存储和加载模型。

## 环境准备

### 1. Python 环境 (仅用于模型转换)
需要安装以下库来运行转换脚本：

```bash
pip install torch numpy gguf asteroid-filterbanks
```

### 2. C++ 编译环境
- CMake
- GCC 或 Clang 编译器

---

## 第一步：准备模型

你需要从 HuggingFace 下载 PyTorch 原始权重并将其转换为 GGUF 格式。

**1. 下载 PyTorch 权重文件**
请下载 `pytorch_model.bin` 文件：
- 分割模型: `pyannote/speaker-diarization-community-1` (在 `segmentation/` 子目录下)
- Embedding 模型: `pyannote/speaker-diarization-community-1` (在 `embedding/` 子目录下)

**2. 转换为 GGUF**

使用 `examples/pyannote/python/` 目录下的脚本进行转换：

```bash
# 1. 转换分割模型 (Segmentation)
python examples/pyannote/python/convert_pyannote_to_ggml.py \
  /path/to/segmentation/pytorch_model.bin \
  examples/pyannote/pyannote-segmentation.gguf

# 2. 转换嵌入模型 (Embedding)
python examples/pyannote/python/convert_embedding_to_ggml.py \
  /path/to/embedding/pytorch_model.bin \
  examples/pyannote/pyannote-embedding.gguf
```

---

## 第二步：编译

在项目根目录下执行标准 CMake 编译流程：

```bash
mkdir build
cd build
cmake ..
make -j4 pyannote-diarization
```

---

## 第三步：运行推理

运行编译生成的二进制文件。请注意，输入音频**必须是 16kHz 采样率的 WAV 文件**。

**命令格式:** 
```bash
./bin/pyannote-diarization <分割模型.gguf> <嵌入模型.gguf> <输入音频.wav>
```

**运行示例:** 
```bash
./bin/pyannote-diarization \
  ../examples/pyannote/pyannote-segmentation.gguf \
  ../examples/pyannote/pyannote-embedding.gguf \
  ../samples/jfk.wav
```
