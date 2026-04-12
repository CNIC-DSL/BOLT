import os
import traceback
from easydict import EasyDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from llm_integration import ChatLocalVLLM
except Exception:
    ChatLocalVLLM = None

OPENAI_PIRCE = {
    'gpt-3.5-turbo-0125': {'input_token': 0.5 / 1000000, 'output_token': 1.5 / 1000000},
    'gpt-4-turbo-2024-04-09': {'input_token': 10.0 / 1000000, 'output_token': 30.0 / 1000000},
    'deepseek-v3:671b': {'input_token': 0.0, 'output_token': 0.0},
    'deepseek-v3:671b-gw': {'input_token': 0.0, 'output_token': 0.0},
    'DeepSeek-V3-0324': {'input_token': 0.0, 'output_token': 0.0},
}

# Global token usage tracker
GLOBAL_TOKEN_USAGE = {
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_tokens': 0
}

def get_global_token_usage():
    return GLOBAL_TOKEN_USAGE

def chat_completion_with_backoff(args, messages, temperature=0.0, max_tokens=256):
    api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or 'EMPTY'
    base_url = getattr(args, 'api_base', '') or os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_API_BASE') or ''
    model = getattr(args, 'llm_model_name', '') or os.getenv('LLM_MODEL') or 'gpt-3.5-turbo'

    try:
        if ChatLocalVLLM is not None:
            llm = ChatLocalVLLM(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
                max_retries=5,
            )
        else:
            try:
                llm = ChatOpenAI(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120,
                    max_retries=5,
                )
            except TypeError:
                llm = ChatOpenAI(
                    model=model,
                    openai_api_base=base_url,
                    openai_api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120,
                    max_retries=5,
                )

        lc_messages = []
        for msg in messages:
            if msg.get('role') == 'system':
                lc_messages.append(SystemMessage(content=msg.get('content', '')))
            else:
                lc_messages.append(HumanMessage(content=msg.get('content', '')))

        response = llm.invoke(lc_messages)
        usage_data = response.response_metadata.get('token_usage', {'prompt_tokens': 0, 'completion_tokens': 0})
        model_name = response.response_metadata.get('model_name', model)

        # Update global token usage
        GLOBAL_TOKEN_USAGE['prompt_tokens'] += usage_data.get('prompt_tokens', 0)
        GLOBAL_TOKEN_USAGE['completion_tokens'] += usage_data.get('completion_tokens', 0)
        GLOBAL_TOKEN_USAGE['total_tokens'] += usage_data.get('total_tokens', 0)

        return EasyDict({
            'choices': [{'message': {'content': response.content}}],
            'usage': usage_data,
            'model': model_name,
        })
    except Exception:
        traceback.print_exc()
        return EasyDict({
            'choices': [{'message': {'content': 'LLM_QUERY_ERROR'}}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
            'model': model,
        })
