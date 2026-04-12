GLOBAL_TOKEN_USAGE = {
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_tokens': 0
}

def update_token_usage(response):
    try:
        # Check if response is from LangChain object
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            if usage:
                GLOBAL_TOKEN_USAGE['prompt_tokens'] += usage.get('prompt_tokens', 0)
                GLOBAL_TOKEN_USAGE['completion_tokens'] += usage.get('completion_tokens', 0)
                GLOBAL_TOKEN_USAGE['total_tokens'] += usage.get('total_tokens', 0)
        # Fallback for dictionary response (if applicable)
        elif isinstance(response, dict) and 'token_usage' in response:
            usage = response['token_usage']
            GLOBAL_TOKEN_USAGE['prompt_tokens'] += usage.get('prompt_tokens', 0)
            GLOBAL_TOKEN_USAGE['completion_tokens'] += usage.get('completion_tokens', 0)
            GLOBAL_TOKEN_USAGE['total_tokens'] += usage.get('total_tokens', 0)

    except Exception:
        pass

def get_global_token_usage():
    return GLOBAL_TOKEN_USAGE

def print_token_usage():
    usage = GLOBAL_TOKEN_USAGE
    print("=" * 40)
    print("Global Token Usage Summary:")
    print(f"   Input (Prompt): {usage['prompt_tokens']}")
    print(f"   Output (Completion): {usage['completion_tokens']}")
    print(f"   Total: {usage['total_tokens']}")
    print("=" * 40)
