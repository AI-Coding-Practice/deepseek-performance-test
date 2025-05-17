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
            "什么是人工智能？",
            "解释一下机器学习的基本概念",
            "深度学习与传统机器学习有什么区别？",
            "神经网络的工作原理是什么？",
            "什么是自然语言处理？"
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
                                    full_response += data['response']
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
            
            with tqdm(total=self.duration, desc="Running Load Test") as pbar:
                while time.time() < end_time:
                    # 控制QPS
                    if len(tasks) < self.qps:
                        task = asyncio.create_task(self.make_request(session))
                        tasks.append(task)
                    
                    # 等待一小段时间以控制QPS
                    await asyncio.sleep(1/self.qps)
                    
                    # 更新进度条
                    pbar.update(1)
                    
                    # 清理已完成的任务
                    done_tasks = [t for t in tasks if t.done()]
                    for task in done_tasks:
                        result = await task
                        self.results.append(result)
                        tasks.remove(task)
            
            # 等待所有剩余任务完成
            if tasks:
                remaining_results = await asyncio.gather(*tasks)
                self.results.extend(remaining_results)
        
        return self.results 