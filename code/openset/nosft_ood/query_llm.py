import os
import requests

class LLMQuerier:
    def __init__(self, base_url, model_name):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.is_deepseek = "deepseek" in model_name.lower()

        if  "cstcloud" in base_url.lower():
            self.url = f"{self.base_url}/chat/completions"  # DeepSeek endpoint (without /v1)
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Environment variable OPENAI_API_KEY is not set.")
        elif "openrouter" in base_url.lower():
            if 'deepseek-v3:671b-gw' == model_name.lower():
                self.model_name = 'deepseek/deepseek-chat-v3-0324'
            self.url = f"{self.base_url}/chat/completions"  # DeepSeek endpoint (without /v1)
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("Environment variable OPENROUTER_API_KEY is not set.")
        else:
            self.url = f"{self.base_url}/v1/chat/completions"  # Standard OpenAI-compatible endpoint

    def query_cot_prompt(self, text, labels, similar_exs, similar_labels):
        prompt = self.construct_prompt(text, labels, similar_exs, similar_labels)
        messages = [
            {"role": "system", "content": "You are an expert in text classification with a focus on Out-of-Distribution (OOD) detection."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3
        }

        headers = {"Content-Type": "application/json"}
        if self.is_deepseek:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(self.url, json=payload, headers=headers)
        response.raise_for_status()
        ans = response.json()['choices'][0]['message']['content'].strip()
        return ans

    def query_llm(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
        }

        headers = {"Content-Type": "application/json"}
        if self.is_deepseek:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # response = requests.post(self.url, json=payload, headers=headers)
        response = requests.post(self.url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()



    def construct_prompt(self, query_text, candidate_labels, similar_exs, similar_labels):
        prompt = (
            f"You are a classification model. "
            f"Your task is to classify the following input into exactly one of the given categories. "
            f"If the input does not belong to any category, output: '000.It is OOD'.\n\n"
            f"Candidate categories:\n{', '.join(candidate_labels)}\n\n"
        )
        for ex_text, ex_label in zip(similar_exs, similar_labels):
            prompt += f"Example ({ex_label}): \"{ex_text}\"\n"

        # prompt = prompt[:5000]

        prompt += (
            f"\nNow classify the following text:\n\"{query_text}\"\n\n"
            f"Only output the label. Do not include any explanation or additional text.\n"
            f"Answer:"
        )
        return prompt



    def query_analogy_prompt(self, text, candidate_labels, pred_label, similar_exs, similar_labels):
        # Step 1: 初步分类
        step1_prompt = self.construct_simple_prompt(text, candidate_labels)
        # Step 2: 类比验证确认类别
        step2_prompt = self.construct_category_verification_prompt(
            text, pred_label, similar_exs, similar_labels
        )
        step2_messages = [
            {"role": "system", "content": "You are an expert in text classification with a focus on Out-of-Distribution (OOD) detection."},
            {"role": "user", "content": step1_prompt},
            {"role": "assistant", "content": pred_label},
            {"role": "user", "content": step2_prompt},
        ]
        final_answer = self.query_llm(step2_messages).strip()

        # 如果返回 No，则覆盖为 OOD
        if  final_answer.lower().startswith("no"):
            return "000.It is OOD"
        return pred_label

    def construct_category_verification_prompt(self, query_text, pred_label, similar_exs, similar_labels):
        prompt = (
            f"You are validating whether a given text fits the category \"{pred_label}\" by comparing it to representative examples.\n"
            f"Below are example texts from category \"{pred_label}\":\n"
        )
        for i, (ex_text, ex_label) in enumerate(zip(similar_exs, similar_labels), 1):
            if ex_label == pred_label:
                prompt += f"Example {i}: \"{ex_text}\"\n"

        prompt += (
            f"\nNow consider the following text:\n"
            f"\"{query_text}\"\n\n"
            f"Does this text fit well with the examples of \"{pred_label}\" above? "
            f"Please start by answering Yes or No, and briefly explain your reasoning.\n"
            f"Answer:"
        )
        return prompt


    def query_simple_prompt(self, text, labels):
        prompt = self.construct_simple_prompt(text, labels)
        messages = [
            {"role": "system", "content": "You are an expert in text classification with a focus on Out-of-Distribution (OOD) detection."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            # "temperature": 0.3
        }

        headers = {"Content-Type": "application/json"}
        if self.is_deepseek:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(self.url, json=payload, headers=headers)
        response.raise_for_status()
        ans = response.json()['choices'][0]['message']['content'].strip()
        return ans

    def construct_simple_prompt(self, query_text, candidate_labels):
        prompt = (
            f"You are a classification model. "
            f"Your task is to classify the following input into exactly one of the given categories. "
            f"If the input does not belong to any category, output: '000.It is OOD'.\n\n"
            f"Candidate categories:\n{', '.join(candidate_labels)}\n\n"
            f"Now classify the following text:\n\"{query_text}\"\n\n"
            # f"Only output the label. Do not include any explanation or additional text.\n"
            f"Answer:"
        )
        return prompt




# ==========================
# ✅ 测试主函数
# ==========================
if __name__ == "__main__":
    import sys

    # 可选修改为 DeepSeek/CSTCloud 接口
    # base_url = "https://uni-api.cstcloud.cn/v1"
    # model_name = "deepseek-v3:671b"
    base_url = "https://openrouter.ai/api/v1"
    model_name = "deepseek/deepseek-chat-v3-0324"

    # 或本地 vLLM 服务：
    # base_url = "http://localhost:8000"
    # model_name = "qwen2-7b"

    querier = LLMQuerier(base_url, model_name)

    test_text = "How can I delete my account?"
    candidate_labels = ["1.Account Issues", "2.Security", "000.It is OOD"]
    similar_exs = [
        "I forgot my account password and can't log in.",
        "My account was hacked and I want to reset my credentials.",
    ]
    similar_labels = ["1.Account Issues", "2.Security"]

    print("Sending prompt...")
    try:
        result = querier.query_cot_prompt(test_text, candidate_labels, similar_exs, similar_labels)
        print("\n✅ Model Prediction:", result)
    except Exception as e:
        print(f"\n❌ Error: {e}")