# 本地翻译服务使用指南

## 概述

本地翻译服务将翻译模型下载到本地，完全离线运行，无需网络连接。

## 优势

✓ **完全离线** - 无需网络连接
✓ **稳定可靠** - 不受网络波动影响
✓ **无限制** - 没有API调用限制
✓ **速度快** - 本地GPU加速
✓ **隐私安全** - 数据不离开本地

## 快速开始

### 1. 安装依赖

```bash
pip install transformers torch sentencepiece
```

### 2. 运行脚本

```bash
python generate_chinese_subtitle_multi_trans.py
```

### 3. 选择本地翻译

在翻译方式选择时，选择：
```
3. 本地翻译 (无需网络，首次需下载模型)
```

## 支持的本地翻译模型

### 英文 → 中文

**Helsinki-NLP/opus-mt-en-zh**（推荐）
- 大小：约500MB
- 质量：优秀
- 速度：快
- 适用：日常使用

**Helsinki-NLP/opus-mt-en-ROMANCE-zh**
- 大小：约1.2GB
- 质量：很好
- 速度：中等
- 适用：高质量需求

### 日文 → 中文

**Helsinki-NLP/opus-mt-ja-zh**
- 大小：约500MB
- 质量：优秀
- 速度：快
- 适用：日文翻译

### 多语言支持

使用M2M100模型支持更多语言：

```python
from transformers import pipeline

# 支持多种语言
translator = pipeline("translation", model="facebook/m2m100_418M")
```

## 详细使用步骤

### 方法1：使用多翻译服务版本（推荐）

```bash
python generate_chinese_subtitle_multi_trans.py
```

交互式选择：
1. 输入视频路径
2. 选择模型大小（推荐：turbo）
3. 选择源语言
4. 选择字幕类型（双语/纯中文）
5. **选择翻译方式：3. 本地翻译**

首次运行会自动下载模型（约500MB），之后完全离线使用。

### 方法2：在代码中直接使用

```python
from transformers import pipeline

# 加载翻译模型（只需一次）
print("正在加载本地翻译模型...")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
print("✓ 模型加载完成")

# 翻译文本
text = "Hello, how are you?"
result = translator(text)
chinese_text = result[0]['translation_text']
print(f"原文: {text}")
print(f"译文: {chinese_text}")
```

### 方法3：使用GPU加速

如果有GPU，可以加速翻译：

```python
import torch
from transformers import pipeline

# 指定使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

translator = pipeline(
    "translation", 
    model="Helsinki-NLP/opus-mt-en-zh",
    device=device
)
```

## 模型下载位置

模型会自动下载到：

**Windows:**
```
C:\Users\<用户名>\.cache\huggingface\hub\
```

**Linux/macOS:**
```
~/.cache/huggingface/hub/
```

## 手动下载模型

如果想提前下载模型：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 指定模型
model_name = "Helsinki-NLP/opus-mt-en-zh"

# 下载模型和tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存到本地
save_directory = "./local_models/opus-mt-en-zh"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"✓ 模型已保存到: {save_directory}")
```

使用本地模型：
```python
from transformers import pipeline

# 从本地加载
translator = pipeline("translation", model="./local_models/opus-mt-en-zh")
```

## 性能对比

| 翻译方式 | 速度 | 质量 | 网络 | 推荐场景 |
|---------|------|------|------|---------|
| Google翻译 | 快 | 优秀 | 需要 | 日常使用 |
| 本地翻译（CPU） | 中等 | 良好 | 不需要 | 离线使用 |
| 本地翻译（GPU） | 快 | 良好 | 不需要 | **推荐** |
| 手动翻译 | - | 最好 | 不需要 | 高质量需求 |

## GPU加速配置

### 检查GPU可用性

```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

### 使用GPU加速翻译

```python
import torch
from transformers import pipeline

# 自动选择设备
device = 0 if torch.cuda.is_available() else -1

# 创建翻译器
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-zh",
    device=device
)

# 翻译
result = translator("Hello world")
print(result[0]['translation_text'])
```

## 批量翻译优化

如果需要翻译大量文本，可以批量处理：

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

# 批量翻译
texts = [
    "Hello world",
    "How are you?",
    "Nice to meet you"
]

# 批量翻译（更快）
results = translator(texts, batch_size=8)

for i, result in enumerate(results):
    print(f"{i}: {result['translation_text']}")
```

## 常见问题

### Q: 模型下载需要多长时间？

A: 
- 500MB模型：约5-15分钟（取决于网络）
- 1.2GB模型：约15-30分钟

### Q: 模型占用多少磁盘空间？

A:
- Helsinki-NLP/opus-mt-en-zh：约500MB
- Helsinki-NLP/opus-mt-en-ROMANCE-zh：约1.2GB
- facebook/m2m100_418M：约1.8GB

### Q: 可以更换翻译模型吗？

A: 可以。修改代码中的模型名称：

```python
# 英文→中文
model="Helsinki-NLP/opus-mt-en-zh"

# 日文→中文
model="Helsinki-NLP/opus-mt-ja-zh"

# 多语言
model="facebook/m2m100_418M"
```

### Q: 本地翻译质量如何？

A: 
- Helsinki模型质量很好，适合日常使用
- 对于专业翻译，建议人工校对
- 质量接近Google翻译，但可能不如专业人工翻译

### Q: 可以在多台机器上使用吗？

A: 可以。每台机器需要：
1. 安装依赖：`pip install transformers torch sentencepiece`
2. 首次运行会下载模型
3. 之后完全离线使用

### Q: 如何更新模型？

A: 
1. 删除缓存目录中的旧模型
2. 重新运行脚本，会自动下载最新版本
3. 或手动指定新模型名称

## 高级配置

### 使用量化模型（减少显存）

```python
from transformers import pipeline

# 使用量化模型（更小，更快）
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-zh",
    torch_dtype=torch.float16  # 使用FP16
)
```

### 自定义分词器

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载模型和分词器
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

# 自定义翻译函数
def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 完整示例

### 示例1：单个视频翻译

```bash
python generate_chinese_subtitle_multi_trans.py
```

选择：
1. 视频路径：`/path/to/video.mp4`
2. 模型：`turbo`
3. 源语言：`英文`
4. 字幕类型：`双语字幕`
5. 翻译方式：`3. 本地翻译`

### 示例2：批量翻译

```python
from transformers import pipeline
import whisper

# 加载翻译模型
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

# 加载Whisper模型
model = whisper.load_model("turbo", device="cuda")

# 转录
result = model.transcribe("video.mp4")

# 翻译每个片段
for segment in result['segments']:
    text = segment['text'].strip()
    translated = translator(text)[0]['translation_text']
    print(f"原文: {text}")
    print(f"译文: {translated}")
```

### 示例3：GPU加速翻译

```python
import torch
from transformers import pipeline

# 检查GPU
if torch.cuda.is_available():
    device = 0
    print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = -1
    print("⚠️  使用CPU")

# 创建翻译器
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-zh",
    device=device
)

# 测试翻译
test_text = "Hello, this is a test."
result = translator(test_text)
print(f"原文: {test_text}")
print(f"译文: {result[0]['translation_text']}")
```

## 推荐配置

### 日常使用（推荐）
- 模型：Helsinki-NLP/opus-mt-en-zh
- 设备：GPU（如果可用）
- 批量大小：8
- 适合：大多数场景

### 高质量需求
- 模型：Helsinki-NLP/opus-mt-en-ROMANCE-zh
- 设备：GPU
- 批量大小：4
- 适合：专业翻译

### 离线使用
- 模型：Helsinki-NLP/opus-mt-en-zh
- 设备：CPU或GPU
- 适合：无网络环境

## 总结

本地翻译服务提供了：
- ✓ 完全离线运行
- ✓ 稳定可靠
- ✓ GPU加速支持
- ✓ 无API限制
- ✓ 隐私安全

首次使用需要下载模型（约500MB），之后完全离线使用，非常适合批量处理视频翻译！

立即开始使用：
```bash
python generate_chinese_subtitle_multi_trans.py
# 选择：3. 本地翻译
```
