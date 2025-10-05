# -*- coding: utf-8 -*-
import os
import logging
import torch
import torch.nn as nn
from typing import Optional

from peft import PeftModel  # 仅在需要时使用
from utils import get_best_checkpoint, create_model


class Model(nn.Module):
    """
    统一的分类模型封装：
    - 可选 LoRA 适配器（仅在目录含 adapter_config.json 且未 --disable_peft 时启用）
    - 否则按普通 checkpoint 加载（与 plm_ood 相同）
    - 提供 features() 抽取句向量；forward() 返回 logits
    """

    def __init__(self, args, tokenizer):
        super().__init__()
        self.backbone = getattr(args, "backbone", "")

        # 1) 选择 checkpoint 目录（若未找到 best，就用 model_path）
        ckpt_best = get_best_checkpoint(getattr(args, "checkpoint_path", None))
        ckpt_dir: str = ckpt_best if ckpt_best is not None else getattr(args, "model_path", "")

        # 2) 创建“基座分类模型”（内部会据 num_labels 构造分类头）
        #    注意：这里的 model_path 传入 create_model，用于选择基础权重（如 ./pretrained_models/bert-base-uncased）
        base_model = create_model(
            model_path=getattr(args, "model_path", ""),
            num_labels=getattr(args, "num_labels", 2),
            tokenizer=tokenizer
        )

        # 3) 判定是否启用 PEFT（LoRA）
        adapter_dir = getattr(args, "peft_adapter", "") or getattr(args, "lora_adapter", "")
        use_peft = (
            bool(adapter_dir)
            and os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
            and not getattr(args, "disable_peft", False)
        )

        logging.info(f"[LOAD] backbone={self.backbone}")
        logging.info(f"[LOAD] base(model_path)={getattr(args,'model_path','')}")
        logging.info(f"[LOAD] ckpt_dir(model_path/best)={ckpt_dir}")
        logging.info(f"[LOAD] peft_adapter={adapter_dir} use_peft={use_peft}")
        logging.info(
            f"[CHECK] exists(base)={os.path.exists(getattr(args,'model_path',''))} "
            f"exists(ckpt_dir)={os.path.exists(ckpt_dir)} "
            f"exists(adapter)={os.path.exists(adapter_dir)}"
        )

        # 4) 依据策略加载最终 self.model
        if use_peft:
            logging.info(f"[PEFT] Loading LoRA adapter: {adapter_dir}")
            self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        else:
            self.model = self._load_non_peft(base_model, ckpt_dir)

        # 5) tokenizer 相关 ids（若缺失则补齐）
        if hasattr(self.model, "config"):
            if getattr(self.model.config, "pad_token_id", None) is None:
                self.model.config.pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if getattr(self.model.config, "eos_token_id", None) is None:
                self.model.config.eos_token_id = getattr(tokenizer, "eos_token_id", None)

        self.model.eval()

        # 6) 解析“编码器”和“分类头”（兼容多结构/PEFT 包裹）
        self.llm, self.fc = self._pick_encoder_and_head(self.model)

    # ---------- 内部：非 PEFT 加载逻辑 ----------
    def _load_non_peft(self, base_model: nn.Module, ckpt_dir: str) -> nn.Module:
        """
        没有/禁用 LoRA 时的加载流程：
        - 若 ckpt_dir 含 adapter_config.json，则当作 LoRA 适配器加载（容错场景）
        - 若含 pytorch_model.bin/model.safetensors，优先 from_pretrained(ckpt_dir)，失败再手工 load_state_dict
        - 否则直接返回 base_model
        """
        if ckpt_dir and os.path.isdir(ckpt_dir):
            adapter_cfg = os.path.join(ckpt_dir, "adapter_config.json")
            pt_bin = os.path.join(ckpt_dir, "pytorch_model.bin")
            safetensors_bin = os.path.join(ckpt_dir, "model.safetensors")

            if os.path.exists(adapter_cfg):
                logging.info(f"[PEFT] adapter_config.json found in ckpt_dir -> load as PEFT: {ckpt_dir}")
                return PeftModel.from_pretrained(base_model, ckpt_dir)

            if os.path.exists(pt_bin) or os.path.exists(safetensors_bin):
                logging.info(f"[STATE_DICT] Loading finetuned weights from: {ckpt_dir}")
                try:
                    # 目录结构兼容时，直接交给 transformers 的 from_pretrained
                    return type(base_model).from_pretrained(ckpt_dir)
                except Exception as e:
                    logging.warning(f"[FROM_PRETRAINED FAIL] fallback to load_state_dict: {e}")
                    model = base_model
                    weight_file = safetensors_bin if os.path.exists(safetensors_bin) else pt_bin
                    state = torch.load(weight_file, map_location="cpu")
                    model.load_state_dict(state, strict=False)
                    return model

        logging.info("[BACKBONE ONLY] Use base_model without external checkpoint.")
        return base_model

    # ---------- 内部：挑选编码器与分类头 ----------
    def _pick_encoder_and_head(self, m: nn.Module):
        """
        从 self.model（或其 base_model）中提取：
          - 编码器（bert/roberta/deberta/distilbert/albert/model）
          - 分类头（classifier/score/classification_head/lm_head）
        若都找不到编码器，就用 m 自身；分类头找不到允许为 None。
        """
        bm = getattr(m, "base_model", None)

        def _pick_encoder(obj: Optional[nn.Module]):
            if obj is None:
                return None
            for name in ["bert", "roberta", "deberta", "deberta_v2", "distilbert", "albert", "model"]:
                if hasattr(obj, name):
                    return getattr(obj, name)
            return None

        enc = _pick_encoder(m) or _pick_encoder(bm) or m  # 兜底：用 m 自身
        head = None
        for name in ["classifier", "score", "classification_head", "lm_head"]:
            if hasattr(m, name):
                head = getattr(m, name)
                break

        return enc, head

    # ---------- 特征抽取 ----------
    def features(self, x):
        """
        返回句向量：
        - BERT 等：优先 pooler_output；否则 CLS（h[:,0]）
        - 其他（如 LLaMA encoder）：用最后一个 token（h[:,-1]）
        - 若无 last_hidden_state，则用 hidden_states[-1]
        - 再不行，用输入 id 对 Embedding 做 mean pooling
        """
        outputs = self.llm(**x, output_hidden_states=True, return_dict=True)

        # (1) BERT 有 pooler_output
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output

        # (2) 常见 encoder：last_hidden_state
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            h = outputs.last_hidden_state
            if any(k in self.backbone for k in ["bert", "roberta", "deberta", "distilbert", "albert"]):
                return h[:, 0]  # CLS
            return h[:, -1]     # 其他结构：最后一个 token

        # (3) 分类输出不带 last_hidden_state：从 hidden_states 兜底
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            h = outputs.hidden_states[-1]  # [B, L, H]
            return h[:, 0]                 # CLS

        # (4) 最后兜底：用 Embedding 的 mean pooling
        input_ids = x.get("input_ids", None)
        if input_ids is not None and hasattr(self.llm, "get_input_embeddings"):
            emb = self.llm.get_input_embeddings()(input_ids)  # [B, L, H]
            attn = x.get("attention_mask", None)
            if attn is not None:
                denom = attn.sum(dim=1, keepdim=True).clamp_min(1)
                return (emb * attn.unsqueeze(-1)).sum(dim=1) / denom
            return emb.mean(dim=1)

        raise RuntimeError(
            "features(): cannot derive hidden states; "
            "encoder output lacks pooler_output/last_hidden_state/hidden_states."
        )

    # ---------- 前向 ----------
    def forward(self, x):
        out = self.model(**x, return_dict=True)
        return out.logits
