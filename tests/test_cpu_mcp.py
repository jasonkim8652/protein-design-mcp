import asyncio
import os
import sys
import httpx
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

LLM_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Using a smart model that excels at tool calling
MODEL_NAME = "openai/gpt-oss-120b:free" 
LLM_API_BASE = os.environ.get("OPENROUTER_API_BASE")
async def run_agent(user_prompt: str):
    if not LLM_API_KEY:
        print("Error: Please set the OPENROUTER_API_KEY environment variable.", file=sys.stderr)
        return

    # 1. Define the parameters to start your custom MCP server
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "protein_design_mcp.server"],
        env=os.environ.copy()
    )

    async with AsyncExitStack() as stack:
        print("[Agent] Connecting to local protein-design-mcp server...")
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        
        # Initialize handshake with the server
        await session.initialize()
        print("[Agent] Handshake successful.")

        # 2. Query the server for its available tools
        mcp_tools_response = await session.list_tools()
        tools = mcp_tools_response.tools
        print(f"[Agent] Discovered {len(tools)} tools from server.")

        # 3. Format MCP tools into the OpenRouter/OpenAI Chat Completion tool schema
        openrouter_tools = []
        for tool in tools:
            openrouter_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        # 4. Construct messages payload
        messages = [{"role": "user", "content": user_prompt}]
        print(f"\n[User]: {user_prompt}")

        # 5. Call OpenRouter
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "HTTP-Referer": "http://localhost:3000", # Required by OpenRouter
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": openrouter_tools if openrouter_tools else None
        }

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(
                LLM_API_BASE,
                headers=headers,
                json=payload
            )
            response_json = response.json()
            
            # Extract choice
            choice = response_json["choices"][0]["message"]
            
            # 6. Check if the LLM wants to execute an MCP tool
            if "tool_calls" in choice and choice["tool_calls"]:
                tool_call = choice["tool_calls"][0]
                tool_name = tool_call["function"]["name"]
                # OpenRouter sends arguments as a stringified JSON object
                import json
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                print(f"\n[LLM Decision]: Decided to call local tool '{tool_name}' with args: {tool_args}")
                print(f"[Agent]: Executing '{tool_name}' on your Python MCP server...")
                
                # Execute the tool on your local server implementation
                tool_result = await session.call_tool(tool_name, arguments=tool_args)
                
                # Display the content returned from your code modifications
                print("\n[Server Output Result]:")
                for content_item in tool_result.content:
                    if content_item.type == "text":
                        print(content_item.text)
            else:
                # The model answered directly without needing a tool
                print(f"\n[LLM Response]: {choice['content']}")

if __name__ == "__main__":
    # Give the agent a prompt that deliberately requires one of your server's tools
    prompt = "predict the structure of MKWVTFISLLFLFSSAYS" 
    asyncio.run(run_agent(prompt))