#!/usr/bin/env python3
"""
Whisper视频翻译性能基准测试脚本
用于比较不同优化方案的性能差异

测试项目：
1. CPU vs GPU 基础性能
2. torch.compile 加速效果
3. 不同模型大小的性能对比
4. 多GPU并行性能
"""

import whisper
import os
import sys
import time
import torch
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, video_path, output_dir="benchmark_results"):
        """
        初始化基准测试
        
        参数:
            video_path: 测试视频路径
            output_dir: 结果输出目录
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.results = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查视频文件
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        self.video_name = os.path.basename(video_path)
        self.video_size = os.path.getsize(video_path) / (1024**2)  # MB
        
        print("=" * 60)
        print("Whisper性能基准测试")
        print("=" * 60)
        print(f"测试视频: {self.video_name}")
        print(f"视频大小: {self.video_size:.2f} MB")
        print(f"输出目录: {output_dir}")
        print("=" * 60)
    
    def run_test(self, test_name, model_size, device, use_compile=False, compile_mode="max-autotune"):
        """
        运行单个测试
        
        参数:
            test_name: 测试名称
            model_size: 模型大小
            device: 设备类型
            use_compile: 是否使用torch.compile
            compile_mode: 编译模式
        
        返回:
            测试结果字典
        """
        print(f"\n{'=' * 60}")
        print(f"测试: {test_name}")
        print(f"{'=' * 60}")
        print(f"模型: {model_size}")
        print(f"设备: {device}")
        print(f"torch.compile: {use_compile}")
        if use_compile:
            print(f"编译模式: {compile_mode}")
        
        try:
            # 记录开始时间
            start_time = time.time()
            model_load_start = time.time()
            
            # 加载模型
            print("\n加载模型...")
            model = whisper.load_model(model_size, device=device)
            model_load_time = time.time() - model_load_start
            print(f"✓ 模型加载完成 ({model_load_time:.2f}s)")
            
            # 应用torch.compile
            if use_compile and device == "cuda":
                try:
                    compile_start = time.time()
                    print("\n应用torch.compile...")
                    model.encoder = torch.compile(model.encoder, mode=compile_mode)
                    model.decoder = torch.compile(model.decoder, mode=compile_mode)
                    compile_time = time.time() - compile_start
                    print(f"✓ 模型编译完成 ({compile_time:.2f}s)")
                except Exception as e:
                    print(f"⚠️  torch.compile失败: {e}")
                    use_compile = False
            
            # 预热
            print("\n预热模型...")
            warmup_start = time.time()
            try:
                # 使用短片段预热
                import numpy as np
                mel = np.random.randn(80, 3000).astype(np.float32)
                if device == "cuda":
                    mel = torch.from_numpy(mel).to(device).half()
                else:
                    mel = torch.from_numpy(mel).to(device)
                
                with torch.no_grad():
                    audio_features = model.encoder(mel)
                    _ = model.decoder(torch.tensor([[model.dims.n_text_ctx // 2]]).to(device), audio_features)
                
                warmup_time = time.time() - warmup_start
                print(f"✓ 预热完成 ({warmup_time:.2f}s)")
            except Exception as e:
                print(f"⚠️  预热失败: {e}")
                warmup_time = 0
            
            # 转录
            print("\n开始转录...")
            transcribe_start = time.time()
            
            # 记录GPU内存
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                gpu_memory_before = torch.cuda.memory_allocated() / 1024**2
            
            result = model.transcribe(
                self.video_path,
                word_timestamps=True,
                fp16=(device == "cuda")
            )
            
            transcribe_time = time.time() - transcribe_start
            print(f"✓ 转录完成 ({transcribe_time:.2f}s)")
            
            # 记录GPU内存
            if device == "cuda":
                gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024**2
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**2
            else:
                gpu_memory_peak = 0
                gpu_memory_after = 0
            
            # 清理
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            
            total_time = time.time() - start_time
            
            # 收集结果
            test_result = {
                "test_name": test_name,
                "model_size": model_size,
                "device": device,
                "use_compile": use_compile,
                "compile_mode": compile_mode if use_compile else None,
                "model_load_time": model_load_time,
                "compile_time": compile_time if use_compile else 0,
                "warmup_time": warmup_time,
                "transcribe_time": transcribe_time,
                "total_time": total_time,
                "gpu_memory_peak_mb": gpu_memory_peak,
                "gpu_memory_after_mb": gpu_memory_after,
                "video_duration": result['segments'][-1]['end'] if result['segments'] else 0,
                "num_segments": len(result['segments']),
                "realtime_factor": transcribe_time / result['segments'][-1]['end'] if result['segments'] else 0
            }
            
            # 显示结果
            print(f"\n{'=' * 60}")
            print(f"测试结果: {test_name}")
            print(f"{'=' * 60}")
            print(f"模型加载时间: {model_load_time:.2f}s")
            if use_compile:
                print(f"编译时间: {compile_time:.2f}s")
            print(f"预热时间: {warmup_time:.2f}s")
            print(f"转录时间: {transcribe_time:.2f}s")
            print(f"总时间: {total_time:.2f}s")
            if device == "cuda":
                print(f"GPU峰值显存: {gpu_memory_peak:.2f} MB")
            print(f"视频时长: {test_result['video_duration']:.2f}s")
            print(f"实时因子: {test_result['realtime_factor']:.2f}x")
            print(f"分段数量: {test_result['num_segments']}")
            
            self.results.append(test_result)
            return test_result
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_tests(self, model_sizes=["tiny", "base", "small"]):
        """
        运行所有测试
        
        参数:
            model_sizes: 要测试的模型大小列表
        """
        print(f"\n开始运行所有测试...")
        print(f"测试模型: {', '.join(model_sizes)}")
        
        # 1. CPU基准测试（仅small模型）
        print(f"\n{'=' * 60}")
        print("1. CPU基准测试")
        print(f"{'=' * 60}")
        self.run_test("CPU-small", "small", "cpu", use_compile=False)
        
        # 2. GPU基准测试
        if torch.cuda.is_available():
            print(f"\n{'=' * 60}")
            print("2. GPU基准测试")
            print(f"{'=' * 60}")
            
            for model_size in model_sizes:
                # 不使用torch.compile
                self.run_test(f"GPU-{model_size}-no-compile", model_size, "cuda", use_compile=False)
                
                # 使用torch.compile（仅small和base）
                if model_size in ["tiny", "base", "small"]:
                    self.run_test(f"GPU-{model_size}-compile-max-autotune", model_size, "cuda", 
                                use_compile=True, compile_mode="max-autotune")
                    self.run_test(f"GPU-{model_size}-compile-reduce-overhead", model_size, "cuda", 
                                use_compile=True, compile_mode="reduce-overhead")
        else:
            print("\n⚠️  CUDA不可用，跳过GPU测试")
        
        # 保存结果
        self.save_results()
        
        # 生成报告
        self.generate_report()
    
    def save_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "video": self.video_name,
                "video_size_mb": self.video_size,
                "timestamp": timestamp,
                "results": self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 结果已保存: {results_file}")
    
    def generate_report(self):
        """生成性能报告"""
        if not self.results:
            print("\n⚠️  没有测试结果，无法生成报告")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 保存CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(self.output_dir, f"benchmark_report_{timestamp}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✓ CSV报告已保存: {csv_file}")
        
        # 生成图表
        self.generate_charts(df)
        
        # 生成文本报告
        self.generate_text_report(df)
    
    def generate_charts(self, df):
        """生成性能图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 转录时间对比
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 按设备分组
        cpu_results = df[df['device'] == 'cpu']
        gpu_results = df[df['device'] == 'cuda']
        
        x_pos = range(len(df))
        bars = ax.bar(x_pos, df['transcribe_time'], 
                     color=['red' if d == 'cpu' else 'blue' for d in df['device']])
        
        ax.set_xlabel('测试')
        ax.set_ylabel('转录时间 (秒)')
        ax.set_title('转录时间对比')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['test_name'], rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        chart_file = os.path.join(self.output_dir, f"transcribe_time_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已保存: {chart_file}")
        
        # 2. 实时因子对比
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(x_pos, df['realtime_factor'],
                     color=['red' if d == 'cpu' else 'blue' for d in df['device']])
        
        ax.set_xlabel('测试')
        ax.set_ylabel('实时因子 (x)')
        ax.set_title('实时因子对比 (越低越好)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['test_name'], rotation=45, ha='right')
        ax.axhline(y=1.0, color='green', linestyle='--', label='实时 (1.0x)')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom')
        
        ax.legend()
        plt.tight_layout()
        chart_file = os.path.join(self.output_dir, f"realtime_factor_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已保存: {chart_file}")
        
        # 3. GPU显存使用对比
        gpu_df = df[df['device'] == 'cuda']
        if not gpu_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x_pos = range(len(gpu_df))
            bars = ax.bar(x_pos, gpu_df['gpu_memory_peak_mb'])
            
            ax.set_xlabel('测试')
            ax.set_ylabel('GPU显存 (MB)')
            ax.set_title('GPU显存使用对比')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(gpu_df['test_name'], rotation=45, ha='right')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}MB',
                       ha='center', va='bottom')
            
            plt.tight_layout()
            chart_file = os.path.join(self.output_dir, f"gpu_memory_{timestamp}.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 图表已保存: {chart_file}")
    
    def generate_text_report(self, df):
        """生成文本报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"benchmark_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Whisper性能基准测试报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"测试视频: {self.video_name}\n")
            f.write(f"视频大小: {self.video_size:.2f} MB\n")
            f.write(f"测试时间: {timestamp}\n")
            f.write(f"测试数量: {len(self.results)}\n\n")
            
            # 详细结果
            f.write("=" * 60 + "\n")
            f.write("详细结果\n")
            f.write("=" * 60 + "\n\n")
            
            for result in self.results:
                f.write(f"测试: {result['test_name']}\n")
                f.write(f"  模型: {result['model_size']}\n")
                f.write(f"  设备: {result['device']}\n")
                f.write(f"  torch.compile: {result['use_compile']}\n")
                if result['use_compile']:
                    f.write(f"  编译模式: {result['compile_mode']}\n")
                f.write(f"  模型加载时间: {result['model_load_time']:.2f}s\n")
                if result['use_compile']:
                    f.write(f"  编译时间: {result['compile_time']:.2f}s\n")
                f.write(f"  预热时间: {result['warmup_time']:.2f}s\n")
                f.write(f"  转录时间: {result['transcribe_time']:.2f}s\n")
                f.write(f"  总时间: {result['total_time']:.2f}s\n")
                if result['device'] == 'cuda':
                    f.write(f"  GPU峰值显存: {result['gpu_memory_peak_mb']:.2f} MB\n")
                f.write(f"  视频时长: {result['video_duration']:.2f}s\n")
                f.write(f"  实时因子: {result['realtime_factor']:.2f}x\n")
                f.write(f"  分段数量: {result['num_segments']}\n")
                f.write("\n")
            
            # 性能对比
            f.write("=" * 60 + "\n")
            f.write("性能对比\n")
            f.write("=" * 60 + "\n\n")
            
            # CPU vs GPU
            cpu_result = df[df['device'] == 'cpu'].iloc[0] if not df[df['device'] == 'cpu'].empty else None
            gpu_results = df[df['device'] == 'cuda']
            
            if cpu_result is not None and not gpu_results.empty:
                f.write("CPU vs GPU对比 (small模型):\n")
                f.write(f"  CPU转录时间: {cpu_result['transcribe_time']:.2f}s\n")
                for _, gpu_result in gpu_results.iterrows():
                    if gpu_result['model_size'] == 'small':
                        speedup = cpu_result['transcribe_time'] / gpu_result['transcribe_time']
                        f.write(f"  {gpu_result['test_name']}: {gpu_result['transcribe_time']:.2f}s (加速: {speedup:.2f}x)\n")
                f.write("\n")
            
            # torch.compile效果
            compile_results = df[df['use_compile'] == True]
            no_compile_results = df[df['use_compile'] == False]
            
            if not compile_results.empty and not no_compile_results.empty:
                f.write("torch.compile加速效果:\n")
                for _, compile_result in compile_results.iterrows():
                    model_size = compile_result['model_size']
                    no_compile = no_compile_results[
                        (no_compile_results['model_size'] == model_size) & 
                        (no_compile_results['device'] == 'cuda')
                    ]
                    if not no_compile.empty:
                        no_compile_time = no_compile.iloc[0]['transcribe_time']
                        compile_time = compile_result['transcribe_time']
                        speedup = no_compile_time / compile_time
                        f.write(f"  {model_size}模型:\n")
                        f.write(f"    无编译: {no_compile_time:.2f}s\n")
                        f.write(f"    有编译 ({compile_result['compile_mode']}): {compile_time:.2f}s\n")
                        f.write(f"    加速: {speedup:.2f}x\n")
                f.write("\n")
            
            # 推荐
            f.write("=" * 60 + "\n")
            f.write("推荐配置\n")
            f.write("=" * 60 + "\n\n")
            
            if not gpu_results.empty:
                # 找到最快的GPU配置
                fastest_gpu = gpu_results.loc[gpu_results['transcribe_factor'].idxmin()]
                f.write(f"最快GPU配置: {fastest_gpu['test_name']}\n")
                f.write(f"  转录时间: {fastest_gpu['transcribe_time']:.2f}s\n")
                f.write(f"  实时因子: {fastest_gpu['realtime_factor']:.2f}x\n\n")
            
            if cpu_result is not None:
                f.write(f"CPU配置: {cpu_result['test_name']}\n")
                f.write(f"  转录时间: {cpu_result['transcribe_time']:.2f}s\n")
                f.write(f"  实时因子: {cpu_result['realtime_factor']:.2f}x\n")
        
        print(f"✓ 文本报告已保存: {report_file}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Whisper性能基准测试工具")
    print("=" * 60)
    
    # 检查PyTorch版本
    print(f"\nPyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 输入视频路径
    video_path = input("\n请输入测试视频路径: ").strip()
    
    if not os.path.exists(video_path):
        print(f"✗ 错误: 视频文件不存在: {video_path}")
        return
    
    # 选择测试模型
    print("\n请选择要测试的模型:")
    print("1. tiny")
    print("2. base")
    print("3. small")
    print("4. medium")
    print("5. all (测试所有)")
    
    choice = input("\n请输入选项 (1-5，默认3): ").strip() or "3"
    
    model_map = {
        "1": ["tiny"],
        "2": ["base"],
        "3": ["small"],
        "4": ["medium"],
        "5": ["tiny", "base", "small"]
    }
    
    model_sizes = model_map.get(choice, ["small"])
    
    # 输出目录
    output_dir = input("\n请输入结果输出目录 (默认: benchmark_results): ").strip() or "benchmark_results"
    
    # 运行基准测试
    try:
        benchmark = PerformanceBenchmark(video_path, output_dir)
        benchmark.run_all_tests(model_sizes)
        
        print("\n" + "=" * 60)
        print("基准测试完成!")
        print("=" * 60)
        print(f"\n结果已保存到: {output_dir}")
        print("请查看以下文件:")
        print("  - benchmark_results_*.json (原始数据)")
        print("  - benchmark_report_*.csv (CSV报告)")
        print("  - benchmark_report_*.txt (文本报告)")
        print("  - *.png (性能图表)")
        
    except Exception as e:
        print(f"\n✗ 基准测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
