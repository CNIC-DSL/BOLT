import os
import traceback
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from easydict import EasyDict

OPENAI_PIRCE = {
    "gpt-3.5-turbo-0125": {"input_token": 0.5 / 1000000, "output_token": 1.5 / 1000000},
    "gpt-4-turbo-2024-04-09": {
        "input_token": 10.0 / 1000000,
        "output_token": 30.0 / 1000000,
    },
    "qwen7b": {"input_token": 0.0000001 / 1000000, "output_token": 0.0000001 / 1000000},
    "deepseek-v3:671b-gw": {
        "input_token": 0.0000001 / 1000000,
        "output_token": 0.0000001 / 1000000,
    },
    "DeepSeek-V3-0324": {
        "input_token": 0.0000001 / 1000000,
        "output_token": 0.0000001 / 1000000,
    },
}


def chat_completion_with_backoff(args, messages, temperature=0.0, max_tokens=256):
    try:

        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        if api_key == "EMPTY":
            print("Warning: OPENAI_API_KEY environment variable is not set.")

        llm = ChatOpenAI(
            model=args.llm_model_name,
            openai_api_key=api_key,
            openai_api_base=args.api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=5,
            timeout=120,
        )

        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))

        response = llm.invoke(lc_messages)

        usage_data = response.response_metadata.get(
            "token_usage", {"prompt_tokens": 0, "completion_tokens": 0}
        )

        return EasyDict(
            {
                "choices": [{"message": {"content": response.content}}],
                "usage": usage_data,
                "model": response.response_metadata.get(
                    "model_name", args.llm_model_name
                ),
            }
        )

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"[FATAL ERROR in utils_LLM.py]: An error occurred.")
        traceback.print_exc()
        print("=" * 50 + "\n")

        return EasyDict(
            {
                "choices": [{"message": {"content": "LLM_QUERY_ERROR"}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "model": args.llm_model_name,
            }
        )


if __name__ == "__main__":

    from easydict import EasyDict

    args = EasyDict(
        {
            "llm_model_name": "deepseek-v3:671b",
            "api_base": "https://uni-api.cstcloud.cn/v1",
        }
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, tell me a short joke."},
    ]

    llm_response = chat_completion_with_backoff(args, messages)

    print("--- LLM Response ---")
    print(llm_response["choices"][0]["message"]["content"])
    print("\n--- Usage ---")
    print(llm_response["usage"])