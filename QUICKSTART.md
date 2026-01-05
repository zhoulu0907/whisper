# Whisper GPU加速 - 快速开始指南

## 🚀 5分钟快速开始

### 1. 安装依赖

```bash
# 安装PyTorch (CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装Whisper和其他依赖
pip install openai-whisper googletrans==4.0.0-rc1 matplotlib pandas
```

### 2. 验证GPU

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

如果显示 `CUDA可用: True`，恭喜！你可以使用GPU加速。

### 3. 运行示例

**单文件处理（GPU优化版本）：**
```bash
python generate_chinese_subtitle_gpu_optimized.py
```

按照提示：
1. 选择处理模式（单文件）
2. 输入视频路径
3. 选择模型（推荐：turbo或small）
4. 选择语言（推荐：自动检测）
5. 选择字幕类型（推荐：双语）
6. 启用torch.compile（推荐：是）

**批量处理（多GPU版本）：**
```bash
python generate_chinese_subtitle_multigpu.py
```

适合处理多个视频文件，自动分配到多个GPU。

---

## 📊 性能对比

| 配置 | 10分钟视频处理时间 | 实时因子 |
|------|------------------|---------|
| CPU (small模型) | ~100-150分钟 | 10-15x |
| GPU基础 (small模型) | ~3-5分钟 | 0.3-0.5x |
| GPU+torch.compile (small模型) | ~2-3分钟 | 0.2-0.3x |
| GPU基础 (turbo模型) | ~2-3分钟 | 0.2-0.3x |

**结论：GPU加速可以带来20-50倍的性能提升！**

---

## 🎯 推荐配置

### 日常使用（推荐）
```bash
python generate_chinese_subtitle_gpu_optimized.py
```
- 模型：turbo
- 启用torch.compile：是
- 编译模式：reduce-overhead

### 批量处理
```bash
python generate_chinese_subtitle_multigpu.py
```
- 模型：turbo
- 使用所有可用GPU

### 高质量需求
```bash
python generate_chinese_subtitle_gpu_optimized.py
```
- 模型：medium或large
- 启用torch.compile：是
- 启用word_timestamps：是

---

## 🔧 常见问题快速解决

### CUDA不可用
```bash
# 重新安装PyTorch CUDA版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 显存不足
- 使用更小的模型（tiny/base/small）
- 关闭其他占用GPU的程序

### FFmpeg未找到
- Windows: 下载并添加到PATH
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

---

## 📖 详细文档

查看完整文档：[GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md)

---

## 🧪 性能测试

运行基准测试：
```bash
python benchmark_performance.py
```

生成详细的性能报告和图表。

---

## 💡 提示

1. **首次使用torch.compile会有编译时间（1-5分钟），后续使用会更快**
2. **turbo模型是速度和质量的最佳平衡**
3. **指定语言可以跳过检测，节省时间**
4. **批量处理使用多GPU版本，效率最高**

---

## 📝 文件说明

| 文件 | 用途 |
|------|------|
| [`generate_chinese_subtitle.py`](generate_chinese_subtitle.py) | 原始版本，基础功能 |
| [`generate_chinese_subtitle_gpu_optimized.py`](generate_chinese_subtitle_gpu_optimized.py) | GPU优化版本，推荐使用 |
| [`generate_chinese_subtitle_multigpu.py`](generate_chinese_subtitle_multigpu.py) | 多GPU并行版本，批量处理 |
| [`benchmark_performance.py`](benchmark_performance.py) | 性能基准测试工具 |
| [`GPU_OPTIMIZATION_GUIDE.md`](GPU_OPTIMIZATION_GUIDE.md) | 完整使用指南 |

---

开始享受GPU加速带来的极速体验吧！🚀
