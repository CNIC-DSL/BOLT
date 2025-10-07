import os
import json
import re
import time
import requests

class LLMQuerier:
    def __init__(self, base_url, model_name):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.is_deepseek = "deepseek" in model_name.lower()

        if "cstcloud" in base_url.lower():
            self.url = f"{self.base_url}/chat/completions"
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Environment variable OPENAI_API_KEY is not set.")
            self._headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        elif "openrouter" in base_url.lower():
            if 'deepseek-v3:671b-gw' == model_name.lower():
                self.model_name = 'deepseek/deepseek-chat-v3-0324'
            self.url = f"{self.base_url}/chat/completions"
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("Environment variable OPENROUTER_API_KEY is not set.")
            # OpenRouter 需要标准 Bearer 头；可选加 Referer / X-Title
            self._headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        else:
            self.url = f"{self.base_url}/v1/chat/completions"
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Environment variable OPENAI_API_KEY is not set.")
            self._headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

    def _coerce_json(self, text):
        """从模型返回里尽量提取 JSON 对象。"""
        try:
            return json.loads(text)
        except Exception:
            pass
        # 粗暴地截取第一个花括号块
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # 尝试修正常见的尾随逗号等问题
        fixed = re.sub(r',\s*([}\]])', r'\1', text)
        fixed = fixed.replace('\n', ' ')
        try:
            return json.loads(fixed)
        except Exception:
            return None

    def zero_shot(self, text, pred, candidate_labels, label_samples, threshold=0.6, max_retries=2, timeout=60):
        """
        使用 LLM 对文本进行二次核验：
        1) 先判断现有预测 pred 是否匹配；
        2) 若不匹配，则对所有候选标签逐一打分 [0,1]；
        3) 若最高分 < threshold，则判定 OOD。
        返回:
        {
          "is_ood": bool,
          "final_label": str,            # 若 OOD 则为 "OOD"
          "confidence": float,           # 对最终决策的总体置信度 [0,1]
          "reason": str,                 # 简要原因
          "scores_by_label": {label: score, ...},  # 可用于回溯
          "pred_agree": bool             # 是否同意原始 pred
        }
        """
        if not isinstance(candidate_labels, (list, tuple)) or len(candidate_labels) == 0:
            raise ValueError("candidate_labels must be a non-empty list/tuple.")

        # # 防止超长：最多取前 200 个标签
        # labels = list(candidate_labels)[:200]

        system_prompt = (
            "You are a precise text classifier for open-set detection. "
            "Follow the rules strictly and output JSON only."
        )

        user_prompt = f"""
Task:
- You are given a text, a current predicted label, a set of instances of this label, and the candidate labels.
- Judge whether the current prediction fits the text semantically.
- If not fit, check if the text belong to other labels in the candidate labels. If not in the candidate labels, the prediction is OOD.

Input:
- Text: {text}
- Current prediction (pred): {pred}
- The instances of this label: {label_samples}
- The candidate labels: {candidate_labels}

Output JSON schema (strictly JSON, no prose):
{{
  "pred_agree": true/false,
  "is_ood": true/false,
  "final_label": label, // from candidate_labels or OOD
  "confidence": float,  // overall confidence in decision [0,1]
  "reason": "<one short sentence>"
}}
"""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": 0,
            # 尽量让服务端返回纯 JSON（部分 OpenAI 兼容服务支持）
            "response_format": {"type": "json_object"}
        }

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(self.url, headers=self._headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = self._coerce_json(content)
                if not parsed:
                    raise ValueError("Model did not return valid JSON.")
                # 兜底字段校验与补全
                out = {
                    "pred_agree": bool(parsed.get("pred_agree", False)),
                    "final_label": parsed.get("final_label") or "OOD",
                    "is_ood": bool(parsed.get("is_ood", True)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reason": parsed.get("reason") or ""
                }
                return out
            except Exception as e:
                last_err = e
                # 简单退避
                time.sleep(0.6 * (attempt + 1))

        raise RuntimeError(f"zero_shot LLM call failed after retries: {last_err}")

    def few_shot(
        self,
        text,
        pred,
        candidate_labels,
        label_samples,
        threshold=0.6,
        per_label_cap=3,
        max_retries=2,
        timeout=60
    ):
        """
        Few-shot 判别：给定每个标签的若干示例，要求 LLM 对各候选标签打分并进行 OOD 判定。

        参数：
          - text: 待判别文本
          - pred: 现有预测标签（用于核验/参考）
          - candidate_labels: 候选标签列表
          - label_samples: dict[str, list[str]]，每个标签对应若干示例文本
          - threshold: 选择标签的最低置信阈值；低于该阈值则判为 OOD
          - per_label_cap: 每个标签最多提供的示例条数（避免 prompt 过长）
          - max_retries, timeout: 请求重试、超时控制

        返回(dict)：
        {
          "is_ood": bool,
          "final_label": str,            # 若 OOD 则为 "OOD"
          "confidence": float,           # [0,1]
          "reason": str,                 # 简要原因
          "scores_by_label": {label: score, ...},
          "pred_agree": bool             # 是否同意原始 pred
        }
        """
        if not isinstance(candidate_labels, (list, tuple)) or len(candidate_labels) == 0:
            raise ValueError("candidate_labels must be a non-empty list/tuple.")
        if not isinstance(label_samples, dict) or len(label_samples) == 0:
            raise ValueError("label_samples must be a non-empty dict of {label: [examples]}.")

        # 仅保留候选标签对应的样本，并做长度/数量裁剪
        labels = list(candidate_labels)[:200]
        pruned_examples = {}
        for lab in [pred]:
            exs = label_samples.get(lab, [])
            if isinstance(exs, (list, tuple)):
                # 轻量清洗
                cleaned = []
                for e in exs[:per_label_cap]:
                    if not isinstance(e, str):
                        continue
                    s = e.strip().replace("\n", " ").strip()
                    if s:
                        cleaned.append(s[:512])  # 避免超长
                if cleaned:
                    pruned_examples[lab] = cleaned

        # 少数情况下某标签没有示例也可，LLM 仍能评分；但最好至少有部分标签有示例
        system_prompt = (
            "You are a careful text classifier for open-set detection. "
            "Use the given label exemplars to compare semantics. "
            "Output valid JSON only."
        )

        # 把示例压成一个紧凑的字符串
        exemplars_lines = []
        for lab in labels:
            exs = pruned_examples.get(lab, [])
            if not exs:
                exemplars_lines.append(f"- {lab}: []")
            else:
                exemplars_lines.append(f"- {lab}: " + json.dumps(exs, ensure_ascii=False))

        exemplars_block = "\n".join(exemplars_lines)

        user_prompt = f"""
Task:
- Given a text, a current prediction, and candidate labels with few-shot examples,
  evaluate how well the text semantically matches each candidate label.
- Score EACH candidate label in [0,1] (0=not fit at all, 1=perfect fit).
- Prefer matching by intent/topic semantics, not surface words only.
- Choose the best label as final IF its score >= {threshold}; otherwise mark as OOD.
- Be conservative: if unsure or all label scores are weak (< {threshold}), choose OOD.

Input:
- Text: {text}
- Current prediction (pred): {pred}
- Candidate labels: {labels}
- Few-shot exemplars per label:
{exemplars_block}

Output JSON schema (strictly JSON, no prose):
{{
  "pred_agree": true/false,                      // whether you agree with 'pred'
  "scores_by_label": {{"<label>": float, ...}},  // score every label in candidate_labels
  "final_label": "<label-or-OOD>",
  "is_ood": true/false,
  "confidence": float,                           // overall confidence [0,1]
  "reason": "<one short sentence>"
}}

Decision rules:
1) Compute scores_by_label for ALL labels using the exemplars.
2) Let best = argmax(score), top = max(score).
   If top >= {threshold}, final_label = best, is_ood = false.
   Else final_label = "OOD", is_ood = true.
3) pred_agree is whether pred equals final_label (and not OOD).
4) Keep JSON minimal and valid.
"""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"}
        }

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(self.url, headers=self._headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = self._coerce_json(content)
                if not parsed:
                    raise ValueError("Model did not return valid JSON.")
                out = {
                    "pred_agree": bool(parsed.get("pred_agree", False)),
                    "scores_by_label": parsed.get("scores_by_label") or {},
                    "final_label": parsed.get("final_label") or "OOD",
                    "is_ood": bool(parsed.get("is_ood", True)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reason": parsed.get("reason") or ""
                }
                # 兜底：如没给分，至少补个 pred 的分
                if not out["scores_by_label"]:
                    out["scores_by_label"] = {str(pred): max(out["confidence"], 0.0)}
                return out
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))

        raise RuntimeError(f"few_shot LLM call failed after retries: {last_err}")

if __name__ == '__main__':
    # ====== 初始化 ======
    base_url = "https://openrouter.ai/api/v1"
    model_name = "deepseek/deepseek-chat-v3-0324"
    llm = LLMQuerier(base_url, model_name)

    # ====== 测试数据 ======
    text = "I want to transfer money to my friend using mobile banking."
    pred = "account inquiry"
    candidate_labels = ["money transfer", "loan application", "technical support", "credit card issue"]
    label_samples = {
        "money transfer": [
            "How do I send funds to another account?",
            "Transfer money to a friend via my bank app."
        ],
        "loan application": [
            "What documents are required to apply for a personal loan?"
        ],
        "technical support": [
            "The app keeps crashing when I try to open it."
        ],
        "credit card issue": [
            "My credit card payment was declined and I don't know why."
        ]
    }

    # ====== few-shot 调用 ======
    result_fs = llm.few_shot(
        text=text,
        pred=pred,
        candidate_labels=candidate_labels,
        label_samples=label_samples,
        threshold=0.6,
        per_label_cap=3
    )

    print("\n=== Few-shot LLM Verification Result ===")
    print(f"Text: {text}")
    print(f"Original prediction: {pred}")
    print(f"Final label: {result_fs['final_label']}")
    print(f"Is OOD: {result_fs['is_ood']}")
    print(f"Confidence: {result_fs['confidence']:.2f}")
    print(f"Reason: {result_fs['reason']}")
    print(f"Scores by label: {result_fs['scores_by_label']}")