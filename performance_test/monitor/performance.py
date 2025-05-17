import psutil
import pynvml
import time
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        # 系统资源指标
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        
        # 响应时间指标
        self.first_token_latency = []  # 首Token延迟
        self.avg_latency = []
        
        # 吞吐量指标
        self.tokens_per_second = []
        self.requests_per_second = []
        
        # 错误率指标
        self.error_count = 0
        self.timeout_count = 0
        self.total_requests = 0
        
        # 时间记录
        self.start_time = None
        self.end_time = None
        
        # 初始化GPU监控
        self.has_gpu = False
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.has_gpu = True
                print(f"GPU监控已初始化，检测到 {device_count} 个GPU设备")
        except Exception as e:
            print(f"GPU监控初始化失败: {str(e)}")
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self._collect_metrics()
    
    def stop(self):
        """停止监控"""
        self.end_time = time.time()
        if self.has_gpu:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def _collect_metrics(self):
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.append(cpu_percent)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.percent)
        
        # GPU指标
        if self.has_gpu:
            try:
                # 获取GPU使用率
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                self.gpu_usage.append(float(gpu_util.gpu))
                
                # 获取GPU显存使用率
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_percent = (gpu_memory.used / gpu_memory.total) * 100
                self.gpu_memory.append(float(gpu_memory_percent))
            except Exception as e:
                print(f"获取GPU指标失败: {str(e)}")
                self.gpu_usage.append(0.0)
                self.gpu_memory.append(0.0)
    
    def record_request(self, response_time: float, first_token_latency: float, tokens: int, is_error: bool = False, is_timeout: bool = False):
        """记录请求指标"""
        self.total_requests += 1
        
        # 记录响应时间
        if response_time > 0:
            self.avg_latency.append(response_time)
        
        # 记录首Token延迟
        if first_token_latency is not None and first_token_latency > 0:
            self.first_token_latency.append(first_token_latency)
        
        # 记录错误
        if is_error:
            self.error_count += 1
        if is_timeout:
            self.timeout_count += 1
        
        # 计算吞吐量
        if response_time > 0:
            self.tokens_per_second.append(tokens / response_time)
            self.requests_per_second.append(1 / response_time)
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """获取收集的指标"""
        metrics = {
            # 系统资源指标
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            
            # 响应时间指标
            "first_token_latency": self.first_token_latency,
            "avg_latency": self.avg_latency,
            
            # 吞吐量指标
            "tokens_per_second": self.tokens_per_second,
            "requests_per_second": self.requests_per_second
        }
        
        # 添加GPU指标（如果有）
        if self.has_gpu:
            metrics.update({
                "gpu_usage": self.gpu_usage,
                "gpu_memory": self.gpu_memory
            })
        
        return metrics
    
    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算统计指标"""
        stats = {}
        
        # 系统资源统计
        for metric_name, values in self.get_metrics().items():
            if values:
                stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "max": float(np.max(values)),
                    "min": float(np.min(values)),
                    "std": float(np.std(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99))
                }
        
        # 响应时间统计
        if self.avg_latency:
            stats["latency"] = {
                "avg_latency": float(np.mean(self.avg_latency)),
                "p50_latency": float(np.percentile(self.avg_latency, 50)),
                "p90_latency": float(np.percentile(self.avg_latency, 90)),
                "p95_latency": float(np.percentile(self.avg_latency, 95)),
                "p99_latency": float(np.percentile(self.avg_latency, 99)),
                "max_latency": float(np.max(self.avg_latency)),
                "min_latency": float(np.min(self.avg_latency)),
                "std_latency": float(np.std(self.avg_latency))
            }
            
            # 首Token延迟统计
            if self.first_token_latency:
                stats["first_token_latency"] = {
                    "avg": float(np.mean(self.first_token_latency)),
                    "p50": float(np.percentile(self.first_token_latency, 50)),
                    "p90": float(np.percentile(self.first_token_latency, 90)),
                    "p95": float(np.percentile(self.first_token_latency, 95)),
                    "p99": float(np.percentile(self.first_token_latency, 99)),
                    "max": float(np.max(self.first_token_latency)),
                    "min": float(np.min(self.first_token_latency)),
                    "std": float(np.std(self.first_token_latency))
                }
        
        # 吞吐量统计
        if self.tokens_per_second:
            stats["throughput"] = {
                "avg_tokens_per_second": float(np.mean(self.tokens_per_second)),
                "p50_tokens_per_second": float(np.percentile(self.tokens_per_second, 50)),
                "p90_tokens_per_second": float(np.percentile(self.tokens_per_second, 90)),
                "p95_tokens_per_second": float(np.percentile(self.tokens_per_second, 95)),
                "p99_tokens_per_second": float(np.percentile(self.tokens_per_second, 99)),
                "max_tokens_per_second": float(np.max(self.tokens_per_second)),
                "min_tokens_per_second": float(np.min(self.tokens_per_second)),
                "std_tokens_per_second": float(np.std(self.tokens_per_second))
            }
        
        if self.requests_per_second:
            stats["throughput"].update({
                "avg_requests_per_second": float(np.mean(self.requests_per_second)),
                "p50_requests_per_second": float(np.percentile(self.requests_per_second, 50)),
                "p90_requests_per_second": float(np.percentile(self.requests_per_second, 90)),
                "p95_requests_per_second": float(np.percentile(self.requests_per_second, 95)),
                "p99_requests_per_second": float(np.percentile(self.requests_per_second, 99)),
                "max_requests_per_second": float(np.max(self.requests_per_second)),
                "min_requests_per_second": float(np.min(self.requests_per_second)),
                "std_requests_per_second": float(np.std(self.requests_per_second))
            })
        
        # 错误率统计
        if self.total_requests > 0:
            stats["error_rates"] = {
                "error_rate": float(self.error_count / self.total_requests * 100),
                "timeout_rate": float(self.timeout_count / self.total_requests * 100),
                "total_requests": int(self.total_requests),
                "error_count": int(self.error_count),
                "timeout_count": int(self.timeout_count)
            }
        
        # 测试持续时间
        if self.start_time and self.end_time:
            stats["test_duration"] = {
                "start_time": datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S'),
                "duration_seconds": float(self.end_time - self.start_time)
            }
        
        return stats 