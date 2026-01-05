#!/usr/bin/env python3
"""
视频字幕批量生成工具
支持单个文件或文件夹批量处理
流程：1. Whisper 转录 -> 2. 翻译成中文 -> 3. 生成字幕

进度监控功能：
- 转录阶段：每10秒显示心跳，确认程序仍在运行
- 翻译阶段：每翻译5个片段或每2秒更新进度
- 生成字幕阶段：实时显示进度条和预计剩余时间
- 所有阶段都显示已用时间和预计剩余时间

语言支持：
- 英文 (English)
- 日文 (Japanese)
- 自动检测 (Auto)
"""

import whisper
import os
import sys
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import subprocess
import shutil


def check_ffmpeg():
    """
    检查系统是否安装了 ffmpeg
    
    返回:
        bool: True 如果 ffmpeg 可用，False 否则
    """
    try:
        # 尝试运行 ffmpeg -version
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def format_timestamp(seconds):
    """将秒数格式化为 SRT 时间戳格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_video(video_path, model_size="small", language=None, device=None, time_limit=None):
    """
    使用 Whisper 转录视频
    
    参数:
        video_path: 视频文件路径
        model_size: 模型大小
        language: 语言代码 (None=自动检测, "English", "Japanese" 等)
        device: 计算设备 (None=自动检测, "cuda", "cpu")
        time_limit: 时间限制（秒），None=无限制
    
    返回:
        Whisper 转录结果
    """
    # 检查 ffmpeg 是否可用
    if not check_ffmpeg():
        print("=" * 60)
        print("✗ 错误: 系统中未找到 ffmpeg")
        print("=" * 60)
        print("\nWhisper 需要 ffmpeg 来处理音频文件。")
        print("\n请安装 ffmpeg:")
        if sys.platform == "win32":
            print("  1. 下载 ffmpeg: https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z")
            print("  2. 解压到某个目录，例如: C:\\ffmpeg")
            print("  3. 将 C:\\ffmpeg\\bin 添加到系统环境变量 PATH 中")
            print("  4. 重启终端/命令提示符")
        elif sys.platform == "darwin":
            print("  运行: brew install ffmpeg")
        else:
            print("  运行: sudo apt install ffmpeg  # Ubuntu/Debian")
            print("  或: sudo yum install ffmpeg    # CentOS/RHEL")
        print("\n安装后，请重新运行此程序。")
        print("=" * 60)
        raise FileNotFoundError("ffmpeg 未安装或不在系统 PATH 中")
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print("=" * 60)
        print(f"✗ 错误: 视频文件不存在")
        print("=" * 60)
        print(f"文件路径: {video_path}")
        print("\n请检查:")
        print("  1. 文件路径是否正确")
        print("  2. 文件是否已被移动或删除")
        print("  3. 路径中是否包含特殊字符")
        print("=" * 60)
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 检查文件是否可读
    try:
        file_size = os.path.getsize(video_path)
    except Exception as e:
        print("=" * 60)
        print(f"✗ 错误: 无法读取文件")
        print("=" * 60)
        print(f"文件路径: {video_path}")
        print(f"错误信息: {e}")
        print("\n请检查:")
        print("  1. 文件是否被其他程序占用")
        print("  2. 文件权限是否正确")
        print("  3. 文件是否损坏")
        print("=" * 60)
        raise
    
    print("=" * 60)
    print(f"视频文件: {video_path}")
    print(f"文件大小: {file_size / (1024**3):.2f} GB")
    print(f"模型大小: {model_size}")
    if language:
        print(f"指定语言: {language}")
    else:
        print("语言检测: 自动")
    print("=" * 60)
    
    # 自动检测最佳设备
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"\n✓ 检测到CUDA可用，将使用GPU加速")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print(f"\n⚠️  CUDA不可用，将使用CPU处理（速度较慢）")
    
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n正在加载 Whisper 模型: {model_size}...")
    model = whisper.load_model(model_size, device=device)
    print("✓ 模型加载完成")
    
    # 转录视频
    print(f"\n开始转录音频...")
    print("注意: 处理时间取决于视频长度和模型大小，请耐心等待...")
    print("=" * 60)
    
    # 添加进度监控
    start_time = time.time()
    last_heartbeat_time = start_time
    
    # 由于 Whisper 的 transcribe 不支持进度回调，我们使用多线程来显示心跳
    import threading
    
    stop_heartbeat = threading.Event()
    
    def heartbeat():
        """心跳线程，定期显示程序仍在运行"""
        while not stop_heartbeat.is_set():
            time.sleep(10)  # 每10秒更新一次
            elapsed = time.time() - start_time
            print(f"\r[心跳] 程序运行中... 已用时: {format_time(elapsed)} | 正在处理音频...", end='', flush=True)
    
    # 启动心跳线程
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    
    # 调用 transcribe
    print(f"\n正在处理...")
    
    # 准备转录参数
    transcribe_kwargs = {
        "word_timestamps": True,
        "fp16": device == "cuda"  # 仅在CUDA上使用FP16加速
    }
    if language:
        transcribe_kwargs["language"] = language
    
    # 如果有时间限制，使用线程来限制处理时间
    if time_limit:
        import threading
        result_container = [None]
        exception_container = [None]
        
        def transcribe_with_timeout():
            try:
                result_container[0] = model.transcribe(video_path, **transcribe_kwargs)
            except Exception as e:
                exception_container[0] = e
        
        transcribe_thread = threading.Thread(target=transcribe_with_timeout)
        transcribe_thread.start()
        transcribe_thread.join(timeout=time_limit)
        
        if transcribe_thread.is_alive():
            print(f"\n⚠️  已达到时间限制 ({format_time(time_limit)})，停止处理")
            # 返回部分结果（如果有的话）
            if result_container[0] is not None:
                result = result_container[0]
                # 过滤掉超过时间限制的片段
                result['segments'] = [s for s in result['segments'] if s['end'] <= time_limit]
                if not result['segments']:
                    print("⚠️  未能在时间限制内处理任何音频片段")
            else:
                # 创建一个空结果
                result = {
                    'language': 'unknown',
                    'segments': [],
                    'text': ''
                }
        else:
            if exception_container[0]:
                raise exception_container[0]
            result = result_container[0]
    else:
        result = model.transcribe(video_path, **transcribe_kwargs)
    
    # 停止心跳线程
    stop_heartbeat.set()
    heartbeat_thread.join(timeout=1)
    
    elapsed = time.time() - start_time
    print(f"\n✓ 转录完成! (用时: {format_time(elapsed)})")
    
    if time_limit and result['segments']:
        print(f"⚠️  注意: 仅处理了前 {format_time(result['segments'][-1]['end'])} 的音频")
    
    print(f"检测到的语言: {result['language']}")
    if result['segments']:
        print(f"处理时长: {result['segments'][-1]['end']:.2f} 秒")
    print(f"分段数量: {len(result['segments'])}")
    else:
        print("分段数量: 0 (无有效片段)")
    
    return result


def format_time(seconds):
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# 全局变量用于翻译进度统计
_translation_count = 0
_translation_total = 0
_translation_start_time = 0
_translation_last_update = 0

def translate_to_chinese(text, translator="google", source_lang="en"):
    """
    将文本翻译成中文
    
    参数:
        text: 源文本
        translator: 翻译器类型 ("google", "baidu", "manual")
        source_lang: 源语言代码 ("en", "ja" 等)
    
    返回:
        中文文本
    """
    global _translation_count, _translation_total, _translation_start_time, _translation_last_update
    
    if translator == "manual":
        # 手动翻译 - 返回原文，用户可以手动翻译
        return text
    elif translator == "google":
        # 使用 Google Translate (需要安装 googletrans)
        try:
            from googletrans import Translator
            translator_obj = Translator()
            
            # 显示翻译进度
            _translation_count += 1
            current_time = time.time()
            
            # 每翻译5个片段或每2秒更新一次进度
            if _translation_count % 5 == 0 or (current_time - _translation_last_update >= 2):
                elapsed = current_time - _translation_start_time
                if _translation_total > 0:
                    progress = _translation_count / _translation_total
                    estimated_total = elapsed / progress if progress > 0 else 0
                    remaining = estimated_total - elapsed
                    print(f"\r翻译中: {_translation_count}/{_translation_total} ({progress*100:.1f}%) | 已用时: {format_time(elapsed)} | 预计剩余: {format_time(remaining)}", end='', flush=True)
                _translation_last_update = current_time
            
            result = translator_obj.translate(text, src=source_lang, dest='zh-CN')
            return result.text
        except ImportError:
            print("\n⚠️  警告: googletrans 未安装")
            print("请运行: .venv/bin/pip install googletrans==4.0.0-rc1")
            print("将返回原文，您可以稍后手动翻译")
            return text
        except Exception as e:
            print(f"\n⚠️  翻译失败: {e}")
            return text
    else:
        return text


def generate_bilingual_subtitle(result, output_path, translator="google", source_lang="en"):
    """
    生成双语字幕（源语言 + 中文）
    
    参数:
        result: Whisper 转录结果
        output_path: 输出文件路径
        translator: 翻译器类型
        source_lang: 源语言代码
    """
    global _translation_count, _translation_total, _translation_start_time, _translation_last_update
    
    print(f"\n正在生成双语字幕...")
    print("=" * 60)
    
    total_segments = len(result['segments'])
    start_time = time.time()
    last_update_time = start_time
    
    # 初始化翻译进度统计
    _translation_count = 0
    _translation_total = total_segments
    _translation_start_time = start_time
    _translation_last_update = start_time
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result['segments'], 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            source_text = segment['text'].strip()
            
            # 翻译成中文
            chinese_text = translate_to_chinese(source_text, translator, source_lang)
            
            # 写入双语字幕
            f.write(f"{i}\n{start} --> {end}\n")
            f.write(f"{source_text}\n")
            f.write(f"{chinese_text}\n\n")
            
            # 进度显示已经在 translate_to_chinese 函数中处理
            pass
    
    elapsed = time.time() - start_time
    print(f"\n✓ 双语字幕已保存: {output_path} (用时: {format_time(elapsed)})")


def generate_chinese_only_subtitle(result, output_path, translator="google", source_lang="en"):
    """
    生成纯中文字幕
    
    参数:
        result: Whisper 转录结果
        output_path: 输出文件路径
        translator: 翻译器类型
        source_lang: 源语言代码
    """
    global _translation_count, _translation_total, _translation_start_time, _translation_last_update
    
    print(f"\n正在生成中文字幕...")
    print("=" * 60)
    
    total_segments = len(result['segments'])
    start_time = time.time()
    last_update_time = start_time
    
    # 初始化翻译进度统计
    _translation_count = 0
    _translation_total = total_segments
    _translation_start_time = start_time
    _translation_last_update = start_time
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result['segments'], 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            source_text = segment['text'].strip()
            
            # 翻译成中文
            chinese_text = translate_to_chinese(source_text, translator, source_lang)
            
            # 写入中文字幕
            f.write(f"{i}\n{start} --> {end}\n")
            f.write(f"{chinese_text}\n\n")
            
            # 进度显示已经在 translate_to_chinese 函数中处理
            pass
    
    elapsed = time.time() - start_time
    print(f"\n✓ 中文字幕已保存: {output_path} (用时: {format_time(elapsed)})")


def generate_english_subtitle(result, output_path):
    """
    生成纯英文字幕
    
    参数:
        result: Whisper 转录结果
        output_path: 输出文件路径
    """
    print(f"\n正在生成英文字幕...")
    print("=" * 60)
    
    total_segments = len(result['segments'])
    start_time = time.time()
    last_update_time = start_time
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result['segments'], 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            
            # 显示进度（每处理100个片段或每2秒更新一次）
            current_time = time.time()
            if i % 100 == 0 or (current_time - last_update_time >= 2):
                progress = i / total_segments
                elapsed = current_time - start_time
                if progress > 0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    print(f"\r生成进度: {i}/{total_segments} ({progress*100:.1f}%) | 已用时: {format_time(elapsed)} | 预计剩余: {format_time(remaining)} | {'█' * int(progress * 30):<30}", end='', flush=True)
                last_update_time = current_time
    
    elapsed = time.time() - start_time
    print(f"\n✓ 英文字幕已保存: {output_path} (用时: {format_time(elapsed)})")


def save_transcript(result, output_path):
    """保存转录文本"""
    print(f"\n正在保存转录文本...")
    print("=" * 60)
    
    total_segments = len(result['segments'])
    start_time = time.time()
    last_update_time = start_time
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result['segments'], 1):
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            f.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")
            
            # 显示进度（每处理100个片段或每2秒更新一次）
            current_time = time.time()
            if i % 100 == 0 or (current_time - last_update_time >= 2):
                progress = i / total_segments
                elapsed = current_time - start_time
                if progress > 0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    print(f"\r保存进度: {i}/{total_segments} ({progress*100:.1f}%) | 已用时: {format_time(elapsed)} | 预计剩余: {format_time(remaining)} | {'█' * int(progress * 30):<30}", end='', flush=True)
                last_update_time = current_time
    
    elapsed = time.time() - start_time
    print(f"\n✓ 转录文本已保存: {output_path} (用时: {format_time(elapsed)})")


def process_video_to_chinese_subtitle(
    video_path,
    model_size="small",
    output_dir=None,
    subtitle_type="bilingual",
    translator="google",
    language=None,
    device=None
):
    """
    处理视频，生成中文字幕
    
    参数:
        video_path: 视频文件路径
        model_size: 模型大小
        output_dir: 输出目录
        subtitle_type: 字幕类型 ("bilingual" 双语, "chinese" 纯中文, "english" 纯英文)
        translator: 翻译器类型 ("google", "manual")
        language: 源语言代码 (None=自动检测, "English", "Japanese" 等)
        device: 计算设备 (None=自动检测, "cuda", "cpu")
    """
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"✗ 错误: 文件不存在: {video_path}")
        return None
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 第一步：使用 Whisper 转录
    # 传递 time_limit 参数（如果有的话）
    result = transcribe_video(video_path, model_size, language, device, time_limit)
    
    # 获取检测到的语言
    detected_lang = result['language']
    source_lang = detected_lang
    
    # 第二步：保存源语言字幕
    source_subtitle_path = os.path.join(output_dir, f"{base_name}_{detected_lang.lower()}.srt")
    generate_english_subtitle(result, source_subtitle_path)
    
    # 第三步：根据需要生成中文字幕
    if subtitle_type == "bilingual":
        bilingual_subtitle_path = os.path.join(output_dir, f"{base_name}_bilingual.srt")
        generate_bilingual_subtitle(result, bilingual_subtitle_path, translator, source_lang)
    elif subtitle_type == "chinese":
        chinese_subtitle_path = os.path.join(output_dir, f"{base_name}_chinese.srt")
        generate_chinese_only_subtitle(result, chinese_subtitle_path, translator, source_lang)
    
    # 保存转录文本
    transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    save_transcript(result, transcript_path)
    
    print("\n" + "=" * 60)
    print("所有任务完成!")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {base_name}_{detected_lang.lower()}.srt ({detected_lang}字幕)")
    if subtitle_type == "bilingual":
        print(f"  - {base_name}_bilingual.srt (双语字幕)")
    elif subtitle_type == "chinese":
        print(f"  - {base_name}_chinese.srt (中文字幕)")
    print(f"  - {base_name}_transcript.txt (转录文本)")
    
    return result


def get_video_files(directory):
    """
    获取目录中的所有视频文件
    
    参数:
        directory: 目录路径
    
    返回:
        视频文件路径列表
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
    video_files = []
    
    for file in Path(directory).iterdir():
        # 跳过以"."开头的文件（如 ._filename.mp4）
        if file.is_file() and file.suffix.lower() in video_extensions and not file.name.startswith('.'):
            video_files.append(str(file))
    
    return sorted(video_files)


def batch_process_videos(
    input_dir,
    model_size="small",
    output_dir=None,
    subtitle_type="bilingual",
    translator="google",
    language=None,
    device=None
):
    """
    批量处理目录中的所有视频文件
    
    参数:
        input_dir: 输入目录
        model_size: 模型大小
        output_dir: 输出目录
        subtitle_type: 字幕类型
        translator: 翻译器类型
        language: 源语言代码
        device: 计算设备 (None=自动检测, "cuda", "cpu")
    """
    # 获取所有视频文件
    video_files = get_video_files(input_dir)
    
    if not video_files:
        print(f"✗ 在目录 {input_dir} 中未找到视频文件")
        return
    
    print(f"\n找到 {len(video_files)} 个视频文件:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(video_file)}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    total_files = len(video_files)
    success_count = 0
    failed_count = 0
    
    print("\n" + "=" * 60)
    print("开始批量处理")
    print("=" * 60)
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n\n{'=' * 60}")
        print(f"处理文件 {i}/{total_files}: {os.path.basename(video_path)}")
        print(f"{'=' * 60}\n")
        
        try:
            # 在开始处理前再次检查文件是否存在
            if not os.path.exists(video_path):
                print(f"✗ 错误: 文件不存在: {video_path}")
                print(f"  该文件可能已被移动或删除")
                failed_count += 1
                continue
            
            result = process_video_to_chinese_subtitle(
                video_path=video_path,
                model_size=model_size,
                output_dir=output_dir,
                subtitle_type=subtitle_type,
                translator=translator,
                language=language,
                device=device
            )
            
            if result:
                success_count += 1
                print(f"\n✓ 文件 {i}/{total_files} 处理成功")
            else:
                failed_count += 1
                print(f"\n✗ 文件 {i}/{total_files} 处理失败")
        
        except FileNotFoundError as e:
            failed_count += 1
            print(f"\n✗ 文件 {i}/{total_files} 处理失败: 文件未找到")
            print(f"  错误详情: {e}")
        except subprocess.CalledProcessError as e:
            failed_count += 1
            print(f"\n✗ 文件 {i}/{total_files} 处理失败: 外部程序错误")
            print(f"  这通常是因为 ffmpeg 未能正确处理该文件")
            print(f"  错误详情: {e}")
        except Exception as e:
            failed_count += 1
            print(f"\n✗ 文件 {i}/{total_files} 处理失败: {type(e).__name__}")
            print(f"  错误详情: {e}")
            # 显示完整的错误堆栈，便于调试
            import traceback
            print(f"\n详细错误信息:")
            traceback.print_exc()
    
    # 输出总结
    print("\n\n" + "=" * 60)
    print("批量处理完成!")
    print("=" * 60)
    print(f"\n处理统计:")
    print(f"  总文件数: {total_files}")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  成功率: {success_count/total_files*100:.1f}%")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("视频字幕批量生成工具")
    print("=" * 60)
    
    # 选择处理模式
    print("\n请选择处理模式:")
    print("1. 单个文件处理")
    print("2. 文件夹批量处理")
    
    mode_choice = input("\n请输入选项 (1-2，默认1): ").strip() or "1"
    
    input_path = None
    is_batch = False
    
    if mode_choice == "2":
        # 批量处理模式
        input_dir = input("\n请输入视频文件夹路径 (默认: /Users/kanten/Downloads/test): ").strip() or "/Users/kanten/Downloads/test"
        
        if not os.path.isdir(input_dir):
            print(f"✗ 错误: 目录不存在: {input_dir}")
            return
        
        input_path = input_dir
        is_batch = True
    else:
        # 单文件处理模式
        video_path = input("\n请输入视频文件路径 (默认: /Users/kanten/Downloads/0102/needsrt/ABF-289-uncensored-nyap2p.com.mp4): ").strip() or "/Users/kanten/Downloads/0102/needsrt/ABF-289-uncensored-nyap2p.com.mp4"
        
        if not os.path.isfile(video_path):
            print(f"✗ 错误: 文件不存在: {video_path}")
            return
        
        input_path = video_path
        is_batch = False
        
        # 询问是否限制处理时间（用于测试）
        print("\n是否限制处理时间用于快速测试？")
        print("1. 是，仅处理前5分钟")
        print("2. 否，处理完整视频（默认）")
        
        time_limit_choice = input("\n请输入选项 (1-2，默认2): ").strip() or "2"
        use_time_limit = time_limit_choice == "1"
    
    # 选择模型大小
    print("\n请选择模型大小:")
    print("1. tiny (最快，质量较低)")
    print("2. base (较快，质量中等)")
    print("3. small (推荐，平衡速度和质量)")
    print("4. medium (较慢，质量较高)")
    print("5. large (最慢，质量最高)")
    print("6. turbo (强烈推荐，速度快且质量好)")
    
    model_choice = input("\n请输入选项 (1-6，默认3): ").strip() or "3"
    
    model_map = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large",
        "6": "turbo"
    }
    
    model_size = model_map.get(model_choice, "small")
    
    # 选择源语言
    print("\n请选择源语言:")
    print("1. 英文 (English)")
    print("2. 日文 (Japanese)")
    print("3. 自动检测 (推荐)")
    
    lang_choice = input("\n请输入选项 (1-3，默认3): ").strip() or "3"
    
    lang_map = {
        "1": "English",
        "2": "Japanese",
        "3": None
    }
    
    language = lang_map.get(lang_choice, None)
    
    # 选择字幕类型
    print("\n请选择字幕类型:")
    print("1. 双语字幕 (源语言 + 中文)")
    print("2. 纯中文字幕")
    print("3. 纯源语言字幕")
    
    subtitle_choice = input("\n请输入选项 (1-3，默认1): ").strip() or "1"
    
    subtitle_map = {
        "1": "bilingual",
        "2": "chinese",
        "3": "english"
    }
    
    subtitle_type = subtitle_map.get(subtitle_choice, "bilingual")
    
    # 选择翻译方式
    if subtitle_type in ["bilingual", "chinese"]:
        print("\n请选择翻译方式:")
        print("1. Google Translate (需要安装 googletrans)")
        print("2. 手动翻译 (先生成源语言字幕，您可以手动翻译)")
        
        translator_choice = input("\n请输入选项 (1-2，默认1): ").strip() or "1"
        
        if translator_choice == "2":
            translator = "manual"
        else:
            translator = "google"
            
            # 尝试安装 googletrans
            print("\n检查翻译依赖...")
            try:
                import googletrans
                print("✓ googletrans 已安装")
            except ImportError:
                print("✗ googletrans 未安装")
                install = input("是否现在安装 googletrans? (y/n): ").strip().lower()
                if install == 'y':
                    import subprocess
                    import sys
                    # 使用当前Python环境的pip
                    pip_path = sys.executable.replace("python.exe", "pip.exe") if sys.platform == "win32" else "pip"
                    subprocess.run([
                        pip_path, "install", "googletrans==4.0.0-rc1"
                    ])
                    print("✓ googletrans 安装完成")
    else:
        translator = None
    
    # 设置输出目录
    if is_batch:
        output_dir = input("\n请输入输出目录 (默认: 与输入目录相同，按 Enter 使用默认): ").strip()
        if not output_dir:
            output_dir = input_path
    else:
        output_dir = os.path.dirname(input_path)
    
    # 开始处理
    print("\n" + "=" * 60)
    print("开始处理")
    print("=" * 60)
    
    if is_batch:
        # 批量处理
        batch_process_videos(
            input_dir=input_path,
            model_size=model_size,
            output_dir=output_dir,
            subtitle_type=subtitle_type,
            translator=translator,
            language=language
        )
    else:
        # 单文件处理
        result = process_video_to_chinese_subtitle(
            video_path=input_path,
            model_size=model_size,
            output_dir=output_dir,
            subtitle_type=subtitle_type,
            translator=translator,
            language=language,
            device=None  # 自动检测最佳设备
        )
        
        if result:
            print("\n字幕文件已生成!")
            print("\n下一步:")
            print("1. 在视频播放器中加载字幕文件")
            if translator == "manual":
                print("2. 使用翻译工具或手动编辑字幕文件")
            print("3. 如果翻译质量不理想，可以手动调整字幕文件")


if __name__ == "__main__":
    main()
