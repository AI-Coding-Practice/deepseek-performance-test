import asyncio
import argparse
from load_generator.generator import LoadGenerator
from monitor.performance import PerformanceMonitor
from collector.data import DataCollector

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='大模型性能测试工具')
    parser.add_argument('--url', default="http://localhost:8001/api/generate", help='目标服务URL')
    parser.add_argument('--model', default='deepseek-r1:1.5b', help='Ollama模型名称')
    parser.add_argument('--qps', type=int, default=10, help='每秒请求数')
    parser.add_argument('--duration', type=int, default=300, help='测试持续时间（秒）')
    parser.add_argument('--token-length', type=int, default=1000, help='生成token的最大长度')
    parser.add_argument('--output-dir', default='test_results', help='测试结果输出目录')
    
    args = parser.parse_args()
    
    # 初始化组件
    monitor = PerformanceMonitor()
    collector = DataCollector(output_dir=args.output_dir)
    
    load_generator = LoadGenerator(
        target_url=args.url,
        model_name=args.model,
        qps=args.qps,
        token_length=args.token_length,
        duration=args.duration,
        monitor=monitor
    )
    
    # 开始监控
    monitor.start()
    
    # 运行负载测试
    print(f"开始性能测试 - 模型: {args.model}, QPS: {args.qps}, 持续时间: {args.duration}秒")
    results = await load_generator.run()
    
    # 停止监控
    monitor.stop()
    
    # 收集性能指标
    metrics = monitor.get_metrics()
    stats = monitor.calculate_statistics()
    
    # 添加测试参数到stats中
    stats.update({
        "model_name": args.model,
        "qps": args.qps,
        "duration": args.duration,
        "token_length": args.token_length
    })
    
    # 保存结果
    collector.save_results(results, metrics, stats)
    
    # 打印关键性能指标
    print("\n性能测试结果:")
    print(f"总请求数: {stats['error_rates']['total_requests']}")
    print(f"成功请求数: {stats['error_rates']['total_requests'] - stats['error_rates']['error_count']}")
    print(f"错误率: {stats['error_rates']['error_rate']:.2f}%")
    print(f"超时率: {stats['error_rates']['timeout_rate']:.2f}%")
    
    print("\n响应时间统计:")
    print(f"平均响应时间: {stats['latency']['avg_latency']:.3f}秒")
    print(f"P50响应时间: {stats['latency']['p50_latency']:.3f}秒")
    print(f"P90响应时间: {stats['latency']['p90_latency']:.3f}秒")
    print(f"P95响应时间: {stats['latency']['p95_latency']:.3f}秒")
    print(f"P99响应时间: {stats['latency']['p99_latency']:.3f}秒")
    print(f"最大响应时间: {stats['latency']['max_latency']:.3f}秒")
    print(f"最小响应时间: {stats['latency']['min_latency']:.3f}秒")
    
    if 'first_token_latency' in stats:
        print("\n首Token延迟统计:")
        print(f"平均首Token延迟: {stats['first_token_latency']['avg']:.3f}秒")
        print(f"P50首Token延迟: {stats['first_token_latency']['p50']:.3f}秒")
        print(f"P90首Token延迟: {stats['first_token_latency']['p90']:.3f}秒")
        print(f"P95首Token延迟: {stats['first_token_latency']['p95']:.3f}秒")
        print(f"P99首Token延迟: {stats['first_token_latency']['p99']:.3f}秒")
        print(f"最大首Token延迟: {stats['first_token_latency']['max']:.3f}秒")
        print(f"最小首Token延迟: {stats['first_token_latency']['min']:.3f}秒")
    
    print("\n吞吐量统计:")
    print(f"平均吞吐量: {stats['throughput']['avg_tokens_per_second']:.2f} tokens/s")
    print(f"P50吞吐量: {stats['throughput']['p50_tokens_per_second']:.2f} tokens/s")
    print(f"P90吞吐量: {stats['throughput']['p90_tokens_per_second']:.2f} tokens/s")
    print(f"P95吞吐量: {stats['throughput']['p95_tokens_per_second']:.2f} tokens/s")
    print(f"P99吞吐量: {stats['throughput']['p99_tokens_per_second']:.2f} tokens/s")
    print(f"最大吞吐量: {stats['throughput']['max_tokens_per_second']:.2f} tokens/s")
    print(f"最小吞吐量: {stats['throughput']['min_tokens_per_second']:.2f} tokens/s")
    
    if 'avg_requests_per_second' in stats['throughput']:
        print("\n请求速率统计:")
        print(f"平均请求速率: {stats['throughput']['avg_requests_per_second']:.2f} requests/s")
        print(f"P50请求速率: {stats['throughput']['p50_requests_per_second']:.2f} requests/s")
        print(f"P90请求速率: {stats['throughput']['p90_requests_per_second']:.2f} requests/s")
        print(f"P95请求速率: {stats['throughput']['p95_requests_per_second']:.2f} requests/s")
        print(f"P99请求速率: {stats['throughput']['p99_requests_per_second']:.2f} requests/s")
        print(f"最大请求速率: {stats['throughput']['max_requests_per_second']:.2f} requests/s")
        print(f"最小请求速率: {stats['throughput']['min_requests_per_second']:.2f} requests/s")
    
    print("\n系统资源使用:")
    print(f"CPU使用率 - 平均: {stats['cpu_usage']['mean']:.2f}%, 最大: {stats['cpu_usage']['max']:.2f}%")
    print(f"内存使用率 - 平均: {stats['memory_usage']['mean']:.2f}%, 最大: {stats['memory_usage']['max']:.2f}%")
    
    if 'gpu_usage' in stats:
        print(f"GPU使用率 - 平均: {stats['gpu_usage']['mean']:.2f}%, 最大: {stats['gpu_usage']['max']:.2f}%")
        print(f"GPU显存使用率 - 平均: {stats['gpu_memory']['mean']:.2f}%, 最大: {stats['gpu_memory']['max']:.2f}%")
    
    print(f"\n测试结果已保存到: {collector.results_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 