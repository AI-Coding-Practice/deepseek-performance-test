# 大模型性能测试框架

这是一个完全用AI实现的项目。

该项目用于测试服务器在处理deepseek等大模型推理任务时的性能的测试框架，支持负载测试、性能监控和数据收集。

## 功能特点

- 支持动态调整QPS（10-500）
- 支持自定义输入Token长度（1000-4096）
- 实时监控系统资源使用情况（CPU、内存、GPU）
- 自动收集和保存测试数据
- 生成详细的测试报告
- 支持首Token延迟统计
- 支持吞吐量分析
- 支持请求速率统计

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd deepseek-performance-test
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

基本用法：
```bash
python performance_test/main.py --url http://your-model-service-url --model deepseek-r1:1.5b --qps 10 --duration 300
```

参数说明：
- `--url`: 目标服务URL（默认：http://localhost:8001/api/generate）
- `--model`: Ollama模型名称（默认：deepseek-r1:1.5b）
- `--qps`: 每秒请求数（默认：10）
- `--duration`: 测试持续时间，单位秒（默认：300）
- `--token-length`: 生成token的最大长度（默认：1000）
- `--output-dir`: 测试结果输出目录（默认：test_results）

## 测试结果

测试结果将保存在指定的输出目录中，包含以下目录：
- `performance_reports`
- `test_results`

测试报告包含以下关键指标：
- 请求统计（总数、成功数、错误率、超时率）
- 响应时间统计（平均、P50/P90/P95/P99、最大/最小）
- 首Token延迟统计（平均、P50/P90/P95/P99、最大/最小）
- 吞吐量统计（tokens/s）
- 请求速率统计（requests/s）
- 系统资源使用情况（CPU、内存、GPU使用率）

## 注意事项

1. 确保目标服务支持POST请求，并接受JSON格式的输入
2. 建议在测试前先进行小规模测试，确保服务正常运行
3. 监控系统资源使用情况，避免过度负载
4. 测试结果会自动保存，请确保有足够的磁盘空间

## 系统要求

- Python 3.8+
- 支持异步IO的操作系统
- 如果使用GPU监控功能，需要安装NVIDIA驱动和CUDA 