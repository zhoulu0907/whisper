# 翻译失败问题解决方案

## 问题描述

运行时出现以下错误：
```
⚠️  翻译失败: [Errno 11001] getaddrinfo failed
```

## 原因分析

错误 `[Errno 11001] getaddrinfo failed` 表示DNS解析失败，无法连接到Google Translate服务器。

可能的原因：
1. **网络连接问题** - 无法访问Google服务
2. **DNS解析失败** - 无法解析translate.google.com
3. **防火墙/代理问题** - 被防火墙阻止
4. **Google Translate服务不可用** - 服务暂时宕机
5. **googletrans库问题** - 库版本不兼容或API已变更

## 解决方案

### 方案1：检查网络连接（最常见）

**测试网络连接：**
```bash
# 测试是否能访问Google
ping translate.google.com

# 或使用curl测试
curl -I https://translate.google.com
```

**如果无法访问：**
- 检查网络连接
- 检查DNS设置
- 尝试使用VPN或代理
- 更换网络环境

### 方案2：使用其他翻译服务

#### 选项A：使用百度翻译

安装百度翻译API：
```bash
pip install baidu-trans-api
```

修改代码使用百度翻译：
```python
from baidu_trans_api import BaiduTrans

# 初始化
translator = BaiduTrans(appid='你的appid', secretKey='你的密钥')

# 翻译
result = translator.translate(text, from='en', to='zh')
```

#### 选项B：使用有道翻译

安装有道翻译：
```bash
pip install youdao-translate
```

#### 选项C：使用DeepL翻译

安装DeepL：
```bash
pip install deepl
```

### 方案3：使用本地翻译模型（推荐）

#### 使用OpenNMT或MarianMT

安装transformers：
```bash
pip install transformers torch sentencepiece
```

使用本地模型翻译：
```python
from transformers import pipeline

# 加载翻译模型
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

# 翻译
result = translator(text)
chinese_text = result[0]['translation_text']
```

**优点：**
- 无需网络连接
- 完全本地运行
- 不受Google服务限制
- 可以离线使用

**缺点：**
- 首次需要下载模型（~500MB-2GB）
- 翻译质量可能不如Google

### 方案4：禁用翻译（快速解决）

如果只需要源语言字幕，可以禁用翻译：

```bash
python generate_chinese_subtitle_fixed.py
```

选择：
- 字幕类型：3. 纯源语言字幕
- 这样就不会尝试翻译

### 方案5：手动翻译（最可靠）

先生成源语言字幕，然后手动翻译：

```bash
python generate_chinese_subtitle_fixed.py
```

选择：
- 字幕类型：3. 纯源语言字幕
- 或：2. 手动翻译

然后使用专业翻译工具：
- Subtitle Edit（Windows）
- Aegisub（跨平台）
- 在线翻译服务

### 方案6：修复googletrans库

#### 更新到最新版本

```bash
pip uninstall googletrans
pip install googletrans==4.0.0-rc1
```

#### 使用fork版本（推荐）

googletrans的官方版本经常出问题，建议使用维护的fork版本：

```bash
pip uninstall googletrans
pip install googletrans-py
```

然后修改代码：
```python
# 原代码
from googletrans import Translator

# 改为
from googletrans_py import Translator
```

#### 使用备用API

修改翻译函数，使用备用API：

```python
def translate_to_chinese(text, translator="google", source_lang="en"):
    if translator == "google":
        try:
            # 方法1：使用备用API
            from googletrans import Translator
            translator_obj = Translator(service_urls=['translate.googleapis.com'])
            result = translator_obj.translate(text, src=source_lang, dest='zh-CN')
            return result.text
        except Exception as e:
            print(f"⚠️  Google翻译失败: {e}")
            # 方法2：返回原文，手动翻译
            return text
```

## 推荐方案

### 方案A：使用本地翻译模型（推荐用于大量翻译）

```python
from transformers import pipeline

# 初始化翻译器（只需一次）
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

def translate_to_chinese(text, translator="local", source_lang="en"):
    if translator == "local":
        try:
            result = translator(text)
            return result[0]['translation_text']
        except Exception as e:
            print(f"⚠️  本地翻译失败: {e}")
            return text
    else:
        # 使用其他翻译服务
        pass
```

**优点：**
- 无需网络
- 速度快
- 可离线使用
- 不受服务限制

### 方案B：使用手动翻译（推荐用于少量视频）

1. 生成源语言字幕
2. 使用专业工具翻译：
   - Subtitle Edit（Windows）- 免费且强大
   - Aegisub（跨平台）- 专业字幕编辑
   - 在线翻译服务

### 方案C：使用百度翻译（推荐用于中文环境）

百度翻译在中国更稳定：

```bash
pip install baidu-trans-api
```

注册百度翻译开放平台获取API密钥：
https://fanyi-api.baidu.com/

## 修改代码以支持多种翻译服务

创建一个新的翻译函数：

```python
def translate_to_chinese(text, translator="google", source_lang="en", translator_config=None):
    """
    支持多种翻译服务的翻译函数
    
    参数:
        text: 源文本
        translator: 翻译器类型 ("google", "baidu", "local", "manual")
        source_lang: 源语言代码
        translator_config: 翻译器配置（如API密钥）
    """
    if translator == "manual":
        return text
    
    elif translator == "google":
        try:
            from googletrans import Translator
            translator_obj = Translator(service_urls=['translate.googleapis.com'])
            result = translator_obj.translate(text, src=source_lang, dest='zh-CN')
            return result.text
        except Exception as e:
            print(f"⚠️  Google翻译失败: {e}")
            return text
    
    elif translator == "baidu":
        try:
            from baidu_trans_api import BaiduTrans
            appid = translator_config.get('appid', '')
            secretKey = translator_config.get('secretKey', '')
            translator_obj = BaiduTrans(appid=appid, secretKey=secretKey)
            result = translator_obj.translate(text, from='en', to='zh')
            return result
        except Exception as e:
            print(f"⚠️  百度翻译失败: {e}")
            return text
    
    elif translator == "local":
        try:
            from transformers import pipeline
            if not hasattr(translate_to_chinese, 'local_translator'):
                translate_to_chinese.local_translator = pipeline(
                    "translation", 
                    model="Helsinki-NLP/opus-mt-en-zh"
                )
            result = translate_to_chinese.local_translator(text)
            return result[0]['translation_text']
        except Exception as e:
            print(f"⚠️  本地翻译失败: {e}")
            return text
    
    return text
```

## 快速测试翻译服务

运行以下代码测试翻译服务：

```python
# 测试Google翻译
try:
    from googletrans import Translator
    t = Translator()
    result = t.translate("Hello", src='en', dest='zh-CN')
    print(f"✓ Google翻译可用: {result.text}")
except Exception as e:
    print(f"✗ Google翻译失败: {e}")

# 测试本地翻译
try:
    from transformers import pipeline
    t = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
    result = t("Hello")
    print(f"✓ 本地翻译可用: {result[0]['translation_text']}")
except Exception as e:
    print(f"✗ 本地翻译失败: {e}")
```

## 总结

| 方案 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| 修复网络 | 最简单 | 可能无法解决 | 网络问题明确 |
| 本地翻译 | 无需网络，稳定 | 需要下载模型 | 大量翻译，离线使用 |
| 手动翻译 | 质量最高 | 耗时 | 少量视频，高质量需求 |
| 百度翻译 | 国内稳定 | 需要API密钥 | 中文环境 |
| 禁用翻译 | 快速 | 无翻译 | 只需源语言字幕 |

## 立即解决

**最快方案：**
```bash
# 选择纯源语言字幕，跳过翻译
python generate_chinese_subtitle_fixed.py
# 字幕类型选择：3. 纯源语言字幕
```

**推荐方案：**
1. 先生成源语言字幕
2. 使用Subtitle Edit等专业工具翻译
3. 这样可以保证翻译质量，避免网络问题
