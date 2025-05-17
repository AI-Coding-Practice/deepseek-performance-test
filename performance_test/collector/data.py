import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataCollector:
    def __init__(self, output_dir: str = "test_results"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"test_{self.timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 定义CSV文件的字段名
        self.fieldnames = [
            "timestamp",
            "response_time",
            "first_token_latency",
            "status",
            "prompt",
            "response",
            "model",
            "prompt_tokens",
            "response_tokens",
            "total_tokens",
            "tokens_per_second",
            "error"  # 添加error字段
        ]
    
    def _convert_numpy_types(self, obj):
        """将NumPy数据类型转换为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj
    
    def save_results(self, load_test_results: List[Dict[str, Any]],
                    performance_metrics: Dict[str, List[float]],
                    performance_stats: Dict[str, Dict[str, float]]):
        """保存所有测试结果"""
        self._save_load_test_results(load_test_results)
        self._save_performance_metrics(performance_metrics)
        self._save_performance_stats(performance_stats)
        self._save_model_responses(load_test_results)
        self._generate_report(load_test_results, performance_metrics, performance_stats)
        self._generate_charts(performance_metrics, performance_stats)
    
    def _save_load_test_results(self, results: List[Dict[str, Any]]):
        """保存负载测试结果到CSV文件"""
        csv_path = os.path.join(self.results_dir, "load_test_results.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            
            # 确保每个结果都包含所有字段
            for result in results:
                row = {field: result.get(field, '') for field in self.fieldnames}
                writer.writerow(row)
    
    def _save_performance_metrics(self, metrics: Dict[str, List[float]]):
        """保存性能指标到CSV文件"""
        csv_path = os.path.join(self.results_dir, "performance_metrics.csv")
        
        # 将指标数据转换为行格式
        rows = []
        for metric_name, values in metrics.items():
            if isinstance(values, list):
                for i, value in enumerate(values):
                    if i >= len(rows):
                        rows.append({})
                    rows[i][metric_name] = value
        
        if rows:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def _save_performance_stats(self, stats: Dict[str, Dict[str, float]]):
        """保存性能统计到JSON文件"""
        json_path = os.path.join(self.results_dir, "performance_stats.json")
        
        # 转换numpy类型为Python原生类型
        stats = self._convert_numpy_types(stats)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def _save_model_responses(self, results: List[Dict[str, Any]]):
        """保存模型响应到文本文件"""
        responses_dir = os.path.join(self.results_dir, "model_responses")
        os.makedirs(responses_dir, exist_ok=True)
        
        # 保存所有响应
        all_responses_path = os.path.join(responses_dir, "all_responses.txt")
        with open(all_responses_path, 'w', encoding='utf-8') as f:
            for result in results:
                # 安全地获取字段值，如果不存在则使用默认值
                timestamp = result.get('timestamp', 0)
                prompt = result.get('prompt', '')
                response = result.get('response', '')
                response_time = result.get('response_time', 0)
                first_token_latency = result.get('first_token_latency')
                prompt_tokens = result.get('prompt_tokens', 0)
                response_tokens = result.get('response_tokens', 0)
                total_tokens = result.get('total_tokens', 0)
                tokens_per_second = result.get('tokens_per_second', 0)
                error = result.get('error', '')
                
                # 写入响应信息
                f.write(f"时间戳: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n")
                if prompt:
                    f.write(f"提示词: {prompt}\n")
                if response:
                    f.write(f"响应: {response}\n")
                f.write(f"响应时间: {response_time:.2f}秒\n")
                if first_token_latency is not None:
                    f.write(f"首Token延迟: {first_token_latency:.2f}秒\n")
                f.write(f"Token统计: 输入={prompt_tokens}, 输出={response_tokens}, 总计={total_tokens}\n")
                f.write(f"吞吐量: {tokens_per_second:.2f} tokens/s\n")
                if error:
                    f.write(f"错误: {error}\n")
                f.write("-" * 80 + "\n")
        
        # 保存请求统计
        stats_path = os.path.join(responses_dir, "response_stats.txt")
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r.get('status', 0) == 200)
        failed_requests = sum(1 for r in results if r.get('status', 0) != 200)
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"总请求数: {total_requests}\n")
            f.write(f"成功请求数: {successful_requests}\n")
            f.write(f"失败请求数: {failed_requests}\n")
            f.write(f"成功率: {(successful_requests/total_requests*100):.2f}%\n\n")
            
            if failed_requests > 0:
                f.write("失败请求详情:\n")
                for result in results:
                    if result.get('status', 0) != 200:
                        timestamp = result.get('timestamp', 0)
                        status = result.get('status', 0)
                        error = result.get('error', '未知错误')
                        
                        f.write(f"时间戳: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"状态码: {status}\n")
                        f.write(f"错误信息: {error}\n")
                        f.write("-" * 40 + "\n")
    
    def _generate_report(self,
                        load_test_results: List[Dict[str, Any]],
                        performance_metrics: Dict[str, List[float]],
                        performance_stats: Dict[str, Dict[str, float]]):
        """生成测试报告"""
        report = {
            "test_summary": {
                "timestamp": self.timestamp,
                "total_requests": len(load_test_results),
                "successful_requests": sum(1 for r in load_test_results if r['status'] == 200),
                "failed_requests": sum(1 for r in load_test_results if r['status'] != 200),
                "test_duration": performance_stats.get('test_duration', {}).get('duration_seconds', 0)
            },
            "performance_metrics": self._convert_numpy_types(performance_metrics),
            "performance_stats": self._convert_numpy_types(performance_stats)
        }
        
        report_path = os.path.join(self.results_dir, "test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    def _generate_charts(self, metrics: Dict[str, List[float]], stats: Dict[str, Dict[str, float]]):
        """生成性能指标图表"""
        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "系统资源使用率", "响应时间分布",
                "吞吐量趋势", "首Token延迟分布",
                "请求统计", "错误率统计"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )
        
        # 1. 系统资源使用率
        if 'cpu_usage' in metrics:
            fig.add_trace(
                go.Scatter(y=metrics['cpu_usage'], name="CPU使用率", line=dict(color='red')),
                row=1, col=1
            )
        if 'memory_usage' in metrics:
            fig.add_trace(
                go.Scatter(y=metrics['memory_usage'], name="内存使用率", line=dict(color='blue')),
                row=1, col=1
            )
        if 'gpu_usage' in metrics:
            fig.add_trace(
                go.Scatter(y=metrics['gpu_usage'], name="GPU使用率", line=dict(color='green')),
                row=1, col=1
            )
        if 'gpu_memory' in metrics:
            fig.add_trace(
                go.Scatter(y=metrics['gpu_memory'], name="GPU显存使用率", line=dict(color='purple')),
                row=1, col=1
            )
        
        # 2. 响应时间分布
        if 'avg_latency' in metrics:
            fig.add_trace(
                go.Box(y=metrics['avg_latency'], name="响应时间", boxpoints='all'),
                row=1, col=2
            )
        
        # 3. 吞吐量趋势
        if 'tokens_per_second' in metrics:
            fig.add_trace(
                go.Scatter(y=metrics['tokens_per_second'], name="Tokens/s", line=dict(color='orange')),
                row=2, col=1
            )
        if 'requests_per_second' in metrics:
            fig.add_trace(
                go.Scatter(y=metrics['requests_per_second'], name="Requests/s", line=dict(color='brown')),
                row=2, col=1
            )
        
        # 4. 首Token延迟分布
        if 'first_token_latency' in metrics:
            fig.add_trace(
                go.Box(y=metrics['first_token_latency'], name="首Token延迟", boxpoints='all'),
                row=2, col=2
            )
        
        # 5. 请求统计
        if 'error_rates' in stats:
            fig.add_trace(
                go.Bar(
                    x=['总请求', '成功请求', '失败请求', '超时请求'],
                    y=[
                        stats['error_rates']['total_requests'],
                        stats['error_rates']['total_requests'] - stats['error_rates']['error_count'],
                        stats['error_rates']['error_count'],
                        stats['error_rates']['timeout_count']
                    ],
                    name="请求统计"
                ),
                row=3, col=1
            )
        
        # 6. 错误率统计
        if 'error_rates' in stats:
            fig.add_trace(
                go.Pie(
                    labels=['成功', '错误', '超时'],
                    values=[
                        stats['error_rates']['total_requests'] - stats['error_rates']['error_count'],
                        stats['error_rates']['error_count'],
                        stats['error_rates']['timeout_count']
                    ],
                    name="错误率统计"
                ),
                row=3, col=2
            )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="性能测试结果可视化",
            showlegend=True
        )
        
        # 保存图表
        html_path = os.path.join(self.results_dir, "performance_charts.html")
        fig.write_html(html_path)
        
        # 生成HTML报告
        self._generate_html_report(metrics, stats, html_path)
    
    def _generate_html_report(self, metrics: Dict[str, List[float]], stats: Dict[str, Dict[str, float]], charts_path: str):
        """生成HTML格式的测试报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>性能测试报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; }}
                .metric-card {{ 
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 10px;
                    border-radius: 5px;
                    display: inline-block;
                    width: 200px;
                }}
                .metric-value {{ 
                    font-size: 24px;
                    font-weight: bold;
                    color: #2196F3;
                }}
                .metric-label {{ 
                    color: #666;
                    margin-top: 5px;
                }}
                .metric-subvalue {{
                    font-size: 16px;
                    color: #666;
                    margin-top: 5px;
                }}
                .resource-card {{
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 10px;
                    border-radius: 5px;
                    display: inline-block;
                    width: 300px;
                }}
                .resource-title {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 10px;
                }}
                .resource-value {{
                    font-size: 16px;
                    color: #666;
                    margin: 5px 0;
                }}
                iframe {{ 
                    width: 100%;
                    height: 1200px;
                    border: none;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>性能测试报告</h1>
                <div class="section">
                    <h2>测试概览</h2>
                    <div class="metric-card">
                        <div class="metric-value">{stats['error_rates']['total_requests']}</div>
                        <div class="metric-label">总请求数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{stats['error_rates']['error_rate']:.2f}%</div>
                        <div class="metric-label">错误率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{stats['error_rates']['timeout_rate']:.2f}%</div>
                        <div class="metric-label">超时率</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>性能指标</h2>
                    <div class="metric-card">
                        <div class="metric-value">{stats['latency']['avg_latency']:.2f}s</div>
                        <div class="metric-label">平均响应时间</div>
                        <div class="metric-subvalue">P99: {stats['latency']['p99_latency']:.2f}s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{stats['throughput']['avg_tokens_per_second']:.2f}</div>
                        <div class="metric-label">平均吞吐量(tokens/s)</div>
                        <div class="metric-subvalue">最大: {stats['throughput']['max_tokens_per_second']:.2f}</div>
                    </div>
                    {f'''
                    <div class="metric-card">
                        <div class="metric-value">{stats['first_token_latency']['avg']:.2f}s</div>
                        <div class="metric-label">平均首Token延迟</div>
                        <div class="metric-subvalue">P99: {stats['first_token_latency']['p99']:.2f}s</div>
                    </div>
                    ''' if 'first_token_latency' in stats else ''}
                </div>
                
                <div class="section">
                    <h2>系统资源使用</h2>
                    <div class="resource-card">
                        <div class="resource-title">CPU使用率</div>
                        <div class="resource-value">平均值: {stats['cpu_usage']['mean']:.2f}%</div>
                        <div class="resource-value">最大值: {stats['cpu_usage']['max']:.2f}%</div>
                        <div class="resource-value">标准差: {stats['cpu_usage']['std']:.2f}%</div>
                    </div>
                    <div class="resource-card">
                        <div class="resource-title">内存使用率</div>
                        <div class="resource-value">平均值: {stats['memory_usage']['mean']:.2f}%</div>
                        <div class="resource-value">最大值: {stats['memory_usage']['max']:.2f}%</div>
                        <div class="resource-value">标准差: {stats['memory_usage']['std']:.2f}%</div>
                    </div>
                    {f'''
                    <div class="resource-card">
                        <div class="resource-title">GPU使用率</div>
                        <div class="resource-value">平均值: {stats['gpu_usage']['mean']:.2f}%</div>
                        <div class="resource-value">最大值: {stats['gpu_usage']['max']:.2f}%</div>
                        <div class="resource-value">标准差: {stats['gpu_usage']['std']:.2f}%</div>
                    </div>
                    <div class="resource-card">
                        <div class="resource-title">GPU显存使用率</div>
                        <div class="resource-value">平均值: {stats['gpu_memory']['mean']:.2f}%</div>
                        <div class="resource-value">最大值: {stats['gpu_memory']['max']:.2f}%</div>
                        <div class="resource-value">标准差: {stats['gpu_memory']['std']:.2f}%</div>
                    </div>
                    ''' if 'gpu_usage' in stats else ''}
                </div>
                
                <div class="section">
                    <h2>详细图表</h2>
                    <iframe src="{os.path.basename(charts_path)}"></iframe>
                </div>
            </div>
        </body>
        </html>
        """
        
        report_path = os.path.join(self.results_dir, "test_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content) 