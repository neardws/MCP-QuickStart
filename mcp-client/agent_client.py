import asyncio
import sys
from openai import OpenAI  # 直接使用 OpenAI 客户端
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import Any, Dict, Callable, List, Optional
import os
from dotenv import load_dotenv
import json

load_dotenv()  # 从 .env 加载环境变量

# DeepSeek LLM API wrapper
class DeepSeekLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 初始化 DeepSeek 客户端，使用 OpenAI 兼容接口
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict:
        """
        调用 DeepSeek API 进行对话
        
        Args:
            messages: 对话历史，格式为 [{"role": "user", "content": "..."}, ...]
            tools: 可选的工具列表
            
        Returns:
            Dict: 返回 LLM 的回复，格式兼容 openai-agents 的 Agent
        """
        try:
            # 使用 OpenAI 兼容接口调用 DeepSeek API
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools if tools else None,
                stream=False
            )
            
            # 处理响应
            message = response.choices[0].message
            result = {
                "role": "assistant",
                "content": message.content
            }
            
            # 如果有工具调用，添加到结果中，并确保转换为字典格式
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # 将 tool_calls 对象转换为字典
                tool_calls_dicts = []
                for tc in message.tool_calls:
                    # 如果已经是字典，直接使用
                    if isinstance(tc, dict):
                        tool_calls_dicts.append(tc)
                    # 否则，转换为字典
                    else:
                        tool_calls_dicts.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        })
                result["tool_calls"] = tool_calls_dicts
            
            return result
            
        except Exception as e:
            print(f"DeepSeek API 调用失败: {str(e)}")
            return {
                "role": "assistant",
                "content": f"抱歉，调用 DeepSeek API 时出错: {str(e)}"
            }

class MCPAgentClient:
    def __init__(self, llm_backend: str = "deepseek"):
        self.llm_backend = llm_backend
        self.llm = self._init_llm()
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    def _init_llm(self):
        if self.llm_backend == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("请设置 DEEPSEEK_API_KEY 环境变量")
            return DeepSeekLLM(api_key)
        # 默认用 openai-agents 的 Agent
        return None

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        response = await self.session.list_tools()
        self.tools = response.tools
        print("\n已连接 MCP Server，工具列表:", [tool.name for tool in self.tools])

    def _make_tools(self):
        # 不再使用 agents.tool.Tool，而是直接创建 OpenAI 格式的工具定义
        openai_tools = []
        for mcp_tool_def in self.tools:
            # 创建 OpenAI 格式的工具定义
            tool_def = {
                "type": "function",
                "function": {
                    "name": mcp_tool_def.name,
                    "description": mcp_tool_def.description,
                    "parameters": mcp_tool_def.inputSchema or {"type": "object", "properties": {}}
                }
            }
            openai_tools.append(tool_def)
        return openai_tools

    async def chat_loop(self):
        print("\nMCP Agent Client Started!")
        print("输入你的问题，输入 'quit' 退出。")
        tools = self._make_tools()
        
        # 创建 OpenAI 客户端（如果使用 OpenAI 后端）
        openai_client = None
        if self.llm_backend == "openai":
            openai_client = OpenAI()
        elif self.llm_backend == "deepseek":
            openai_client = self.llm
        
        # 消息历史
        messages = []
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                
                # 添加用户消息
                messages.append({"role": "user", "content": query})
                
                # 选择 LLM 后端
                if self.llm_backend == "deepseek":
                    # 使用 DeepSeek
                    response = self.llm.chat(messages, tools=tools)
                    assistant_message = response["content"]
                    print("\n" + assistant_message)
                    
                    # 处理工具调用
                    if "tool_calls" in response:
                        for tool_call in response["tool_calls"]:
                            # 检查 tool_call 的类型并相应地访问其属性
                            if isinstance(tool_call, dict):
                                # 如果是字典，直接使用下标访问
                                tool_name = tool_call["function"]["name"]
                                tool_args = json.loads(tool_call["function"]["arguments"])
                                tool_call_id = tool_call["id"]
                            else:
                                # 如果是对象，使用属性访问
                                tool_name = tool_call.function.name
                                tool_args = json.loads(tool_call.function.arguments)
                                tool_call_id = tool_call.id
                                # 转换为字典以便存储在消息历史中
                                tool_call = tool_call.model_dump()
                            
                            print(f"Calling tool: {tool_name} with args: {tool_args}")
                            result = await self.session.call_tool(tool_name, tool_args)
                            
                            # 确保工具结果是字符串
                            result_content = result.content
                            if not isinstance(result_content, str):
                                # 如果结果不是字符串，尝试转换为字符串
                                if hasattr(result_content, 'text'):
                                    # 如果是 TextContent 对象
                                    result_content = result_content.text
                                elif isinstance(result_content, list) and all(hasattr(item, 'text') for item in result_content):
                                    # 如果是 TextContent 对象列表
                                    result_content = "\n".join(item.text for item in result_content)
                                else:
                                    # 其他情况，尝试 JSON 序列化
                                    try:
                                        result_content = json.dumps(result_content)
                                    except:
                                        result_content = str(result_content)
                            
                            print(f"Tool result: {result_content}")
                            
                            # 添加工具结果到消息历史
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": result_content  # 使用处理后的字符串
                            })
                            
                            # 获取 LLM 对工具结果的回应
                            follow_up = self.llm.chat(messages, tools=tools)
                            print("\n" + follow_up["content"])
                            messages.append({"role": "assistant", "content": follow_up["content"]})
                    else:
                        # 添加助手回复到消息历史
                        messages.append({"role": "assistant", "content": assistant_message})
                        
                elif openai_client:
                    # 使用 OpenAI
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        tools=tools
                    )
                    
                    # 处理回复
                    assistant_message = response.choices[0].message
                    print("\n" + (assistant_message.content or ""))
                    
                    # 处理工具调用
                    if assistant_message.tool_calls:
                        for tool_call in assistant_message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)
                            
                            print(f"Calling tool: {tool_name} with args: {tool_args}")
                            result = await self.session.call_tool(tool_name, tool_args)
                            
                            # 确保工具结果是字符串
                            result_content = result.content
                            if not isinstance(result_content, str):
                                # 如果结果不是字符串，尝试转换为字符串
                                if hasattr(result_content, 'text'):
                                    # 如果是 TextContent 对象
                                    result_content = result_content.text
                                elif isinstance(result_content, list) and all(hasattr(item, 'text') for item in result_content):
                                    # 如果是 TextContent 对象列表
                                    result_content = "\n".join(item.text for item in result_content)
                                else:
                                    # 其他情况，尝试 JSON 序列化
                                    try:
                                        result_content = json.dumps(result_content)
                                    except:
                                        result_content = str(result_content)
                            
                            print(f"Tool result: {result_content}")
                            
                            # 添加工具结果到消息历史
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call.model_dump()]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_content  # 使用处理后的字符串
                            })
                            
                            # 获取 LLM 对工具结果的回应
                            follow_up = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=messages
                            )
                            follow_up_content = follow_up.choices[0].message.content
                            print("\n" + follow_up_content)
                            messages.append({"role": "assistant", "content": follow_up_content})
                    else:
                        # 添加助手回复到消息历史
                        messages.append(assistant_message.model_dump())
                else:
                    print("\nError: No valid LLM backend configured")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    backend = "deepseek"
    server_script = "/Users/neardws/Documents/GitHub/MCP-QuickStart/weather/weather.py"
    client = MCPAgentClient(llm_backend=backend)
    print(f"Using {backend} as LLM backend")
    try:
        await client.connect_to_server(server_script)
        print("Connected to server")
        print("Starting chat loop")
        await client.chat_loop()
        print("Chat loop ended")
    finally:
        print("Cleaning up")
        await client.cleanup()
        print("Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main()) 