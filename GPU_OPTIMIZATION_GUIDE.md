# Whisper视频翻译工具 - GPU优化使用指南

## 目录

1. [概述](#概述)
2. [版本对比](#版本对比)
3. [环境要求](#环境要求)
4. [安装说明](#安装说明)
5. [使用方法](#使用方法)
6. [性能优化技巧](#性能优化技巧)
7. [故障排除](#故障排除)
8. [性能基准测试](#性能基准测试)

---

## 概述

本项目提供了三个版本的Whisper视频翻译工具，每个版本都针对不同的使用场景进行了GPU加速优化：

### 版本说明

1. **原始版本** (`generate_chinese_subtitle.py`)
   - 基础功能，支持单文件和批量处理
   - 自动检测GPU，使用FP16加速
   - 适合日常使用

2. **GPU优化版本** (`generate_chinese_subtitle_gpu_optimized.py`)
   - 集成torch.compile加速（PyTorch 2.0+）
   - 增强的GPU内存管理
   - 实时GPU性能监控
   - 适合追求极致性能的用户

3. **多GPU并行版本** (`generate_chinese_subtitle_multigpu.py`)
   - 支持多GPU并行处理
   - 自动负载均衡
   - 适合批量处理大量视频

---

## 版本对比

| 特性 | 原始版本 | GPU优化版本 | 多GPU并行版本 |
|------|---------|-----------|-------------|
| GPU加速 | ✓ FP16 | ✓ FP16 + torch.compile | ✓ FP16 |
| 多GPU支持 | ✗ | ✗ | ✓ |
| 性能监控 | 基础 | 详细（显存、时间） | 详细（多GPU） |
| 批量处理 | ✓ | ✓ | ✓（并行） |
| torch.compile | ✗ | ✓ | ✗ |
| 推荐场景 | 日常使用 | 单GPU高性能 | 批量处理 |

---

## 环境要求

### 基础要求

- Python 3.8+
- FFmpeg（用于音频处理）

### GPU要求

- NVIDIA GPU（支持CUDA 11.0+）
- CUDA Toolkit 11.0+
- cuDNN 8.0+
- 显存建议：
  - tiny/base模型：≥4GB
  - small模型：≥6GB
  - medium模型：≥8GB
  - large模型：≥12GB

### PyTorch要求

- PyTorch 2.0+（用于torch.compile）
- 建议安装CUDA版本的PyTorch

---

## 安装说明

### 1. 安装Python依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper
pip install googletrans==4.0.0-rc1
pip install matplotlib pandas
```

### 2. 安装FFmpeg

**Windows:**
1. 下载FFmpeg: https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z
2. 解压到某个目录，例如: `C:\ffmpeg`
3. 将 `C:\ffmpeg\bin` 添加到系统环境变量PATH中
4. 重启终端

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install ffmpeg
```

### 3. 验证安装

```bash
# 检查FFmpeg
ffmpeg -version

# 检查PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 检查Whisper
python -c "import whisper; print('Whisper installed successfully')"
```

---

## 使用方法

### 1. 原始版本（基础使用）

```bash
python generate_chinese_subtitle.py
```

**交互式选项：**
- 选择处理模式（单文件/批量）
- 选择模型大小（tiny/base/small/medium/large/turbo）
- 选择源语言（英文/日文/自动检测）
- 选择字幕类型（双语/纯中文/纯英文）
- 选择翻译方式（Google/手动）

**命令行使用示例：**
```python
from generate_chinese_subtitle import process_video_to_chinese_subtitle

# 处理单个视频
result = process_video_to_chinese_subtitle(
    video_path="path/to/video.mp4",
    model_size="small",
    output_dir="output",
    subtitle_type="bilingual",
    translator="google",
    language=None,  # 自动检测
    device="cuda"   # 使用GPU
)
```

### 2. GPU优化版本（推荐）

```bash
python generate_chinese_subtitle_gpu_optimized.py
```

**新增选项：**
- 启用/禁用torch.compile加速
- 选择编译模式：
  - `max-autotune`: 最快，编译时间较长
  - `reduce-overhead`: 平衡速度和编译时间
  - `default`: 默认模式

**性能监控输出示例：**
```
============================================================
性能统计
============================================================
总用时: 2m 30s

GPU信息:
  GPU数量: 1
  GPU型号: NVIDIA RTX 3090
  显存使用:
    已分配: 2048.00 MB
    已保留: 4096.00 MB
    峰值: 6144.00 MB
  速度: 150.00 秒
```

**代码示例：**
```python
from generate_chinese_subtitle_gpu_optimized import process_video_to_chinese_subtitle_optimized

# 使用torch.compile加速
result = process_video_to_chinese_subtitle_optimized(
    video_path="path/to/video.mp4",
    model_size="small",
    use_compile=True,
    compile_mode="max-autotune"
)
```

### 3. 多GPU并行版本（批量处理）

```bash
python generate_chinese_subtitle_multigpu.py
```

**使用场景：**
- 处理大量视频文件
- 拥有多块GPU
- 需要最大化吞吐量

**任务分配示例：**
```
✓ 多GPU处理模式
  可用GPU数量: 2
  使用GPU数量: 2
    GPU 0: NVIDIA RTX 3090
    GPU 1: NVIDIA RTX 3080

任务分配:
  GPU 0: 5 个文件
  GPU 1: 5 个文件
```

**性能提升：**
- 理论加速比：GPU数量
- 实际加速比：GPU数量 × 0.8-0.9（考虑负载均衡开销）

---

## 性能优化技巧

### 1. 模型选择

| 模型 | 参数量 | 显存需求 | 速度 | 质量 | 推荐场景 |
|------|--------|---------|------|------|---------|
| tiny | 39M | ~1GB | 最快 | 较低 | 快速测试 |
| base | 74M | ~1GB | 快 | 中等 | 日常使用 |
| small | 244M | ~2GB | 中等 | 较好 | **推荐** |
| medium | 769M | ~5GB | 较慢 | 好 | 高质量需求 |
| large | 1550M | ~10GB | 慢 | 最好 | 最高质量 |
| turbo | 809M | ~3GB | 快 | 好 | **强烈推荐** |

### 2. torch.compile优化

**启用torch.compile可以带来10-30%的性能提升**

```python
# 最快模式（编译时间较长）
model.encoder = torch.compile(model.encoder, mode="max-autotune")
model.decoder = torch.compile(model.decoder, mode="max-autotune")

# 平衡模式（推荐）
model.encoder = torch.compile(model.encoder, mode="reduce-overhead")
model.decoder = torch.compile(model.decoder, mode="reduce-overhead")
```

**注意事项：**
- 首次运行会有编译时间（1-5分钟）
- 编译后的模型会缓存，后续使用更快
- 需要PyTorch 2.0+

### 3. 内存优化

**清理GPU内存：**
```python
import torch
import gc

# 清理未使用的张量
del model

# 清空CUDA缓存
torch.cuda.empty_cache()

# 触发垃圾回收
gc.collect()
```

**监控GPU内存：**
```python
# 当前显存使用
allocated = torch.cuda.memory_allocated() / 1024**2  # MB
reserved = torch.cuda.memory_reserved() / 1024**2  # MB

# 峰值显存使用
torch.cuda.reset_peak_memory_stats()
# ... 运行模型 ...
peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
```

### 4. 批量处理优化

**单GPU批量处理：**
```python
from generate_chinese_subtitle_gpu_optimized import batch_process_videos_optimized

batch_process_videos_optimized(
    input_dir="path/to/videos",
    model_size="turbo",
    use_compile=True,
    compile_mode="reduce-overhead"
)
```

**多GPU批量处理：**
```python
from generate_chinese_subtitle_multigpu import batch_process_videos_multigpu

batch_process_videos_multigpu(
    input_dir="path/to/videos",
    model_size="turbo",
    num_gpus=2  # 使用2个GPU
)
```

### 5. 其他优化技巧

**指定语言：**
```python
# 指定语言可以跳过语言检测，节省时间
result = model.transcribe(
    video_path,
    language="English",  # 或 "Japanese"
    fp16=True
)
```

**禁用word_timestamps：**
```python
# 如果不需要词级时间戳，可以禁用以提升速度
result = model.transcribe(
    video_path,
    word_timestamps=False  # 默认为False
)
```

**使用更小的采样率：**
```python
# 对于长视频，可以降低采样率以加快处理
result = model.transcribe(
    video_path,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6
)
```

---

## 故障排除

### 1. CUDA不可用

**问题：**
```
✗ CUDA不可用，将使用CPU处理（速度较慢）
```

**解决方案：**
1. 检查NVIDIA驱动是否正确安装
2. 检查CUDA版本是否兼容
3. 重新安装PyTorch CUDA版本：
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### 2. 显存不足

**问题：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
1. 使用更小的模型（tiny/base/small）
2. 减少batch size
3. 清理GPU内存：
   ```python
   torch.cuda.empty_cache()
   ```
4. 关闭其他占用GPU的程序

### 3. torch.compile失败

**问题：**
```
⚠️  torch.compile失败: ...
```

**解决方案：**
1. 确认PyTorch版本 ≥ 2.0
2. 更新PyTorch：
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```
3. 使用不同的编译模式（reduce-overhead或default）

### 4. FFmpeg未找到

**问题：**
```
✗ 错误: 系统中未找到 ffmpeg
```

**解决方案：**
按照[安装说明](#安装说明)安装FFmpeg

### 5. 翻译失败

**问题：**
```
⚠️  翻译失败: ...
```

**解决方案：**
1. 检查网络连接
2. 重新安装googletrans：
   ```bash
   pip uninstall googletrans
   pip install googletrans==4.0.0-rc1
   ```
3. 使用手动翻译模式

---

## 性能基准测试

### 运行基准测试

```bash
python benchmark_performance.py
```

**测试内容：**
1. CPU vs GPU性能对比
2. torch.compile加速效果
3. 不同模型大小的性能
4. GPU显存使用情况

**输出文件：**
- `benchmark_results_*.json` - 原始测试数据
- `benchmark_report_*.csv` - CSV格式报告
- `benchmark_report_*.txt` - 文本格式报告
- `transcribe_time_*.png` - 转录时间对比图
- `realtime_factor_*.png` - 实时因子对比图
- `gpu_memory_*.png` - GPU显存使用对比图

### 性能指标说明

**转录时间（Transcribe Time）：**
- 实际转录音频所需的时间
- 越短越好

**实时因子（Realtime Factor）：**
- 转录时间 / 音频时长
- < 1.0x：快于实时
- 1.0x：实时
- > 1.0x：慢于实时

**GPU显存（GPU Memory）：**
- 峰值显存使用量
- 越低越好

**典型性能参考（RTX 3090, small模型）：**
- CPU：~10-15x（实时因子）
- GPU（无编译）：~0.3-0.5x
- GPU（torch.compile）：~0.2-0.4x

---

## 最佳实践

### 1. 日常使用
- 使用GPU优化版本
- 模型选择：turbo或small
- 启用torch.compile（reduce-overhead模式）

### 2. 批量处理
- 使用多GPU并行版本（如果有多个GPU）
- 模型选择：turbo
- 指定语言（跳过检测）

### 3. 高质量需求
- 模型选择：medium或large
- 使用GPU优化版本
- 启用word_timestamps

### 4. 快速测试
- 模型选择：tiny或base
- 禁用word_timestamps
- 指定语言

---

## 常见问题

**Q: torch.compile能提升多少性能？**
A: 通常可以提升10-30%的性能，具体取决于硬件和模型大小。

**Q: 多GPU版本能提升多少性能？**
A: 理论上是GPU数量的倍数，实际约为GPU数量的0.8-0.9倍。

**Q: 应该选择哪个模型？**
A: 推荐使用turbo模型，它在速度和质量之间取得了很好的平衡。

**Q: 如何监控GPU使用情况？**
A: 使用`nvidia-smi`命令或GPU优化版本内置的性能监控功能。

**Q: 可以在CPU上使用torch.compile吗？**
A: 不建议，torch.compile主要针对GPU优化，在CPU上效果不明显。

---

## 技术支持

如有问题，请检查：
1. 环境是否正确安装
2. GPU驱动和CUDA版本是否兼容
3. 显存是否充足
4. 网络连接是否正常（用于翻译）

---

## 更新日志

### v1.0 (2024)
- 原始版本发布
- GPU优化版本发布
- 多GPU并行版本发布
- 性能基准测试工具发布

---

## 许可证

本项目遵循原Whisper项目的许可证。
