#!/usr/bin/env python3
"""
视频字幕生成脚本
支持为视频生成字幕文件，包括转录和翻译功能
"""

import whisper
import os
import sys
import json
from pathlib import Path


def format_timestamp(seconds):
    """将秒数格式化为 SRT 时间戳格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def save_srt(result, output_path):
    """保存 SRT 格式字幕"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result['segments'], 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"✓ SRT 字幕已保存: {output_path}")


def save_vtt(result, output_path):
    """保存 VTT 格式字幕"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in result['segments']:
            start = format_timestamp(segment['start']).replace(',', '.')
            end = format_timestamp(segment['end']).replace(',', '.')
            text = segment['text'].strip()
            f.write(f"{start} --> {end}\n{text}\n\n")
    print(f"✓ VTT 字幕已保存: {output_path}")


def save_txt(result, output_path):
    """保存纯文本格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"✓ 纯文本已保存: {output_path}")


def save_json(result, output_path):
    """保存 JSON 格式（包含所有详细信息）"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON 已保存: {output_path}")


def process_video(
    video_path,
    model_size="small",
    language=None,
    task="transcribe",
    output_formats=None,
    output_dir=None,
    word_timestamps=True
):
    """
    处理视频并生成字幕
    
    参数:
        video_path: 视频文件路径
        model_size: 模型大小 (tiny, base, small, medium, large, turbo)
        language: 视频语言代码 (如 "Chinese", "English", "Japanese")
        task: 任务类型 ("transcribe" 转录 或 "translate" 翻译成英语)
        output_formats: 输出格式列表，如 ["srt", "vtt", "txt", "json"]
        output_dir: 输出目录（默认与视频同目录）
        word_timestamps: 是否包含单词级时间戳
    """
    
    if output_formats is None:
        output_formats = ["srt", "txt"]
    
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
    
    print("=" * 60)
    print(f"视频文件: {video_path}")
    print(f"文件大小: {os.path.getsize(video_path) / (1024**3):.2f} GB")
    print(f"模型大小: {model_size}")
    print(f"任务类型: {task}")
    if language:
        print(f"指定语言: {language}")
    print("=" * 60)
    
    # 加载模型
    print(f"\n正在加载模型: {model_size}...")
    model = whisper.load_model(model_size)
    print("✓ 模型加载完成")
    
    # 准备转录参数
    kwargs = {
        "word_timestamps": word_timestamps
    }
    if language:
        kwargs["language"] = language
    
    print(f"\n开始处理视频...")
    print("注意: 处理时间取决于视频长度和模型大小，请耐心等待...")
    
    # 转录/翻译
    result = model.transcribe(video_path, task=task, **kwargs)
    
    print(f"\n✓ 处理完成!")
    print(f"检测到的语言: {result['language']}")
    print(f"总时长: {result['segments'][-1]['end']:.2f} 秒")
    print(f"分段数量: {len(result['segments'])}")
    
    # 保存输出文件
    print(f"\n正在保存字幕文件...")
    
    for fmt in output_formats:
        output_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        
        if fmt == "srt":
            save_srt(result, output_path)
        elif fmt == "vtt":
            save_vtt(result, output_path)
        elif fmt == "txt":
            save_txt(result, output_path)
        elif fmt == "json":
            save_json(result, output_path)
    
    print("\n" + "=" * 60)
    print("所有任务完成!")
    print("=" * 60)
    
    return result


def main():
    """主函数"""
    # 视频文件路径
    video_path = "/Users/kanten/Downloads/test/test.mkv"
    
    # 配置参数
    config = {
        "model_size": "small",      # 模型大小: tiny, base, small, medium, large, turbo
        "language": None,           # 语言代码，如 "Chinese", "English", "Japanese"
                                    # 设置为 None 让模型自动检测
        "task": "transcribe",       # 任务类型:
                                    # "transcribe" - 转录（保持原语言）
                                    # "translate" - 翻译成英语（注意：只能翻译成英语）
        "output_formats": ["srt", "txt", "vtt"],  # 输出格式
        "output_dir": "/Users/kanten/Downloads/test",  # 输出目录
        "word_timestamps": True     # 包含单词级时间戳
    }
    
    # 根据视频语言调整配置
    print("\n请选择视频的原始语言:")
    print("1. 中文 (生成中文字幕)")
    print("2. 英语 (生成英文字幕)")
    print("3. 日语 (生成日语字幕)")
    print("4. 其他语言 (需要翻译)")
    print("5. 让模型自动检测语言")
    
    choice = input("\n请输入选项 (1-5): ").strip()
    
    language_map = {
        "1": "Chinese",
        "2": "English",
        "3": "Japanese"
    }
    
    if choice in language_map:
        config["language"] = language_map[choice]
        config["task"] = "transcribe"
        print(f"\n将生成 {language_map[choice]} 字幕")
    elif choice == "4":
        print("\n注意: Whisper 的翻译功能只能翻译成英语")
        print("如果您需要翻译成中文，需要:")
        print("  1. 先用 Whisper 翻译成英语")
        print("  2. 再用其他翻译工具（如 Google Translate API）翻译成中文")
        print("\n现在将翻译成英语...")
        config["task"] = "translate"
    elif choice == "5":
        config["language"] = None
        config["task"] = "transcribe"
        print("\n将自动检测语言并生成字幕")
    else:
        print("\n无效选项，将自动检测语言")
        config["language"] = None
        config["task"] = "transcribe"
    
    # 选择模型大小
    print("\n请选择模型大小:")
    print("1. tiny (最快，质量较低)")
    print("2. base (较快，质量中等)")
    print("3. small (推荐，平衡速度和质量)")
    print("4. medium (较慢，质量较高)")
    print("5. large (最慢，质量最高)")
    print("6. turbo (快速高质量，但不支持翻译)")
    
    model_choice = input("\n请输入选项 (1-6，默认3): ").strip() or "3"
    
    model_map = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large",
        "6": "turbo"
    }
    
    if model_choice in model_map:
        config["model_size"] = model_map[model_choice]
    
    # 如果选择了 turbo 模型但需要翻译，提示用户
    if config["model_size"] == "turbo" and config["task"] == "translate":
        print("\n⚠️  警告: turbo 模型不支持翻译功能!")
        confirm = input("是否切换到 small 模型? (y/n): ").strip().lower()
        if confirm == 'y':
            config["model_size"] = "small"
    
    # 处理视频
    print("\n" + "=" * 60)
    print("开始处理视频")
    print("=" * 60)
    
    result = process_video(**config)
    
    if result:
        print("\n字幕文件已生成，您可以:")
        print(f"1. 在视频播放器中加载字幕文件")
        print(f"2. 使用字幕编辑软件进行编辑")
        print(f"3. 如果需要翻译成中文，可以使用翻译工具处理生成的文本文件")


if __name__ == "__main__":
    main()
