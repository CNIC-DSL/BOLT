import os
import traceback
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from easydict import EasyDict

# --- 保留并更新价格字典 ---
# 我们保留价格计算的部分，并为你自己的模型添加一个条目
# 注意：这个价格只是为了让代码不报错，你可以随便设置
OPENAI_PIRCE = {
    'gpt-3.5-turbo-0125': {'input_token': 0.5 / 1000000, 'output_token': 1.5 / 1000000},
    'gpt-4-turbo-2024-04-09': {'input_token': 10.0 / 1000000, 'output_token': 30.0 / 1000000},
    'qwen7b': {'input_token': 0.0000001 / 1000000, 'output_token': 0.0000001 / 1000000},
    # vvvvv 为你的模型添加条目 vvvvv
    'deepseek-v3:671b-gw': {'input_token': 0.0000001 / 1000000, 'output_token': 0.0000001 / 1000000},
    'DeepSeek-V3-0324': {'input_token': 0.0000001 / 1000000, 'output_token': 0.0000001 / 1000000},
}

# --- 重构核心调用函数 ---
def chat_completion_with_backoff(args, messages, temperature=0.0, max_tokens=256):
    """
    使用 LangChain 调用 LLM，并集成了重试和超时逻辑。
    返回一个与原生openai库相似结构的字典，以保持兼容性。
    """
    try:
        # 从环境变量获取 API Key (这是安全的做法)
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY") # 提供一个默认值以防万一
        if api_key == "EMPTY":
            print("Warning: OPENAI_API_KEY environment variable is not set.")

        # 从 args 动态读取配置
        llm = ChatOpenAI(
            model=args.llm_model_name,      # 从args读取模型名称
            openai_api_key=api_key,
            openai_api_base=args.api_base,  # 从args读取API地址
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=5,                  # 内置的强大重试机制
            timeout=120,                    # 넉넉한 超时时间
        )

        # langchain 需要特定的 Message 对象
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        response = llm.invoke(lc_messages)
        
        # --- 模拟原生openai的返回格式 ---
        # 这样做可以最大限度地减少对 al_manager.py 的修改
        usage_data = response.response_metadata.get('token_usage', {
            'prompt_tokens': 0,
            'completion_tokens': 0
        })

        return EasyDict({
            'choices': [{'message': {'content': response.content}}],
            'usage': usage_data,
            'model': response.response_metadata.get('model_name', args.llm_model_name)
        })

    except Exception as e:
        print("\n" + "="*50)
        print(f"[FATAL ERROR in utils_LLM.py]: An error occurred.")
        traceback.print_exc()
        print("="*50 + "\n")
        # 返回一个包含错误信息的结构，避免程序崩溃
        return EasyDict({
            'choices': [{'message': {'content': 'LLM_QUERY_ERROR'}}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
            'model': args.llm_model_name
        })

# --- 用于独立测试的 main 函数 ---
if __name__ == '__main__':
    
    # 模拟 args 对象
    from easydict import EasyDict
    args = EasyDict({
        "llm_model_name": "deepseek-v3:671b",  # 换成你的模型
        "api_base": "https://uni-api.cstcloud.cn/v1", # 换成你的API地址
        # 确保你的环境变量 OPENAI_API_KEY 已设置
    })

    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": 'Hello, tell me a short joke.'},
    ]

    llm_response = chat_completion_with_backoff(args, messages)
    
    print("--- LLM Response ---")
    print(llm_response['choices'][0]['message']['content'])
    print("\n--- Usage ---")
    print(llm_response['usage'])