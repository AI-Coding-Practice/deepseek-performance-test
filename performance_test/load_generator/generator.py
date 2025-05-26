import asyncio
import random
import time
from typing import List, Dict, Any
import aiohttp
from tqdm import tqdm
import tiktoken  # 用于计算token数量
import json

class LoadGenerator:
    def __init__(self, 
                 target_url: str,
                 model_name: str = "deepseek-coder",
                 qps: int = 10,
                 token_length: int = 1000,
                 duration: int = 300,
                 template: str = None,
                 monitor = None):
        self.target_url = target_url
        self.model_name = model_name
        self.qps = qps
        self.token_length = token_length
        self.duration = duration
        self.template = template or "请回答以下问题：{question}"
        self.results = []
        self.monitor = monitor
        
        # 初始化tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # 使用通用的tokenizer
        except:
            self.tokenizer = None
        
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())  # 简单的回退方案
    
    async def generate_prompt(self) -> str:
        """生成测试用的提示词"""
        questions = [
            # 基础概念类
            "什么是人工智能？",
            "解释一下机器学习的基本概念",
            "深度学习与传统机器学习有什么区别？",
            "神经网络的工作原理是什么？",
            "什么是自然语言处理？",
            
            # 技术细节类
            "请详细解释反向传播算法",
            "什么是注意力机制？它在Transformer中是如何工作的？",
            "解释一下梯度消失和梯度爆炸问题",
            "什么是过拟合？如何防止过拟合？",
            "请解释一下卷积神经网络的结构",
            
            # 应用场景类
            "人工智能在医疗领域有哪些应用？",
            "自动驾驶技术目前面临哪些主要挑战？",
            "如何将机器学习应用到金融风控中？",
            "人工智能在智能制造中的应用场景有哪些？",
            "机器学习在推荐系统中是如何工作的？",
            
            # 编程相关类
            "如何使用Python实现一个简单的神经网络？",
            "解释一下PyTorch和TensorFlow的主要区别",
            "如何优化深度学习模型的训练速度？",
            "什么是数据增强？请给出几个具体的例子",
            "如何评估机器学习模型的性能？",
            
            # 前沿发展类
            "大语言模型的发展趋势是什么？",
            "多模态AI技术目前面临哪些挑战？",
            "量子计算对人工智能发展有什么影响？",
            "联邦学习的主要优势是什么？",
            "AI安全与伦理问题主要包含哪些方面？",
            
            # 实践问题类
            "如何处理不平衡数据集？",
            "如何选择合适的机器学习算法？",
            "特征工程的主要方法有哪些？",
            "如何解释机器学习模型的预测结果？",
            "模型部署时需要注意哪些问题？"
        ]
        question = random.choice(questions)
        return self.template.format(question=question)
    
    async def make_request(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """发送单个请求并记录结果"""
        prompt = await self.generate_prompt()
        start_time = time.time()
        first_token_time = None
        is_error = False
        is_timeout = False
        full_response = ""
        
        try:
            # Ollama API格式 - 启用流式响应
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,  # 启用流式响应
                "options": {
                    "num_predict": self.token_length,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            async with session.post(
                self.target_url,
                json=request_data,
                timeout=300  # 设置5分钟超时
            ) as response:
                async for line in response.content:
                    if line:
                        # 记录第一个token的到达时间
                        if first_token_time is None:
                            first_token_time = time.time()
                        
                        try:
                            chunk = line.decode('utf-8').strip()
                            if chunk.startswith('data: '):
                                chunk = chunk[6:]  # 移除 'data: ' 前缀
                            if chunk:
                                data = json.loads(chunk)
                                if 'response' in data:
                                    token_text = data['response']
                                    full_response += token_text
                                    
                                    # 计算当前token的数量并记录到监控器
                                    if self.monitor:
                                        token_count = self.count_tokens(token_text)
                                        if token_count > 0:
                                            self.monitor.record_token(token_count)
                        except:
                            continue
                
                response_time = time.time() - start_time
                first_token_latency = first_token_time - start_time if first_token_time else None
                
                # 计算token数量
                prompt_tokens = self.count_tokens(prompt)
                response_tokens = self.count_tokens(full_response)
                total_tokens = prompt_tokens + response_tokens
                
                # 记录性能指标
                if self.monitor:
                    self.monitor.record_request(
                        response_time=response_time,
                        first_token_latency=first_token_latency,
                        tokens=total_tokens,
                        is_error=False,
                        is_timeout=False
                    )
                
                return {
                    "timestamp": start_time,
                    "response_time": response_time,
                    "first_token_latency": first_token_latency,
                    "status": response.status,
                    "prompt": prompt,
                    "response": full_response,
                    "model": self.model_name,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "tokens_per_second": total_tokens / response_time if response_time > 0 else 0
                }
        except asyncio.TimeoutError:
            is_timeout = True
            response_time = time.time() - start_time
            if self.monitor:
                self.monitor.record_request(
                    response_time=response_time,
                    first_token_latency=None,
                    tokens=0,
                    is_error=False,
                    is_timeout=True
                )
            return {
                "timestamp": start_time,
                "response_time": response_time,
                "first_token_latency": None,
                "status": 408,
                "error": "Request timeout",
                "model": self.model_name,
                "prompt_tokens": 0,
                "response_tokens": 0,
                "total_tokens": 0,
                "tokens_per_second": 0
            }
        except (aiohttp.ClientError, aiohttp.ServerDisconnectedError) as e:
            is_error = True
            response_time = time.time() - start_time
            if self.monitor:
                self.monitor.record_request(
                    response_time=response_time,
                    first_token_latency=None,
                    tokens=0,
                    is_error=True,
                    is_timeout=False
                )
            return {
                "timestamp": start_time,
                "response_time": response_time,
                "first_token_latency": None,
                "status": 500,
                "error": f"Connection error: {str(e)}",
                "model": self.model_name,
                "prompt_tokens": 0,
                "response_tokens": 0,
                "total_tokens": 0,
                "tokens_per_second": 0
            }
        except Exception as e:
            is_error = True
            response_time = time.time() - start_time
            if self.monitor:
                self.monitor.record_request(
                    response_time=response_time,
                    first_token_latency=None,
                    tokens=0,
                    is_error=True,
                    is_timeout=False
                )
            return {
                "timestamp": start_time,
                "response_time": response_time,
                "first_token_latency": None,
                "status": 500,
                "error": str(e),
                "model": self.model_name,
                "prompt_tokens": 0,
                "response_tokens": 0,
                "total_tokens": 0,
                "tokens_per_second": 0
            }
    
    async def run(self):
        """运行负载测试"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            start_time = time.time()
            end_time = start_time + self.duration
            total_requests = 0
            completed_requests = 0
            total_response_time = 0
            
            # 创建进度条，使用总请求数作为进度
            with tqdm(total=self.qps * self.duration, desc="Running Load Test") as pbar:
                # 第一阶段：创建任务
                while time.time() < end_time:
                    # 控制QPS
                    if len(tasks) < self.qps:
                        task = asyncio.create_task(self.make_request(session))
                        tasks.append(task)
                        total_requests += 1
                        pbar.total = total_requests  # 更新总请求数
                    
                    # 等待一小段时间以控制QPS
                    await asyncio.sleep(1/self.qps)
                    
                    # 清理已完成的任务
                    done_tasks = [t for t in tasks if t.done()]
                    for task in done_tasks:
                        result = await task
                        self.results.append(result)
                        tasks.remove(task)
                        completed_requests += 1
                        if result.get('response_time'):
                            total_response_time += result['response_time']
                        pbar.update(1)  # 更新已完成的任务数
                    
                    # 更新进度条信息
                    status = f"Completed: {completed_requests}/{total_requests}"
                    pbar.set_postfix_str(status)
                
                # 第二阶段：等待剩余任务完成
                if tasks:
                    pbar.set_description("Running Load Test")
                    while tasks:
                        done_tasks = [t for t in tasks if t.done()]
                        for task in done_tasks:
                            result = await task
                            self.results.append(result)
                            tasks.remove(task)
                            completed_requests += 1
                            if result.get('response_time'):
                                total_response_time += result['response_time']
                            pbar.update(1)  # 更新已完成的任务数
                        
                        # 更新进度条信息
                        status = f"Completed: {completed_requests}/{total_requests}"
                        pbar.set_postfix_str(status)
                        await asyncio.sleep(0.1)
        
        return self.results 