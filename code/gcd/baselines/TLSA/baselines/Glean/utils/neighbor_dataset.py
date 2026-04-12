import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from heapq import nlargest
import openai
import time
import traceback
import re
import json
import os
# from together import Together
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def _query_llm_with_langchain(prompt_text, args):
    max_retries = 3
    retry_delay = 2 # seconds

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or args.api_key
    api_base = os.getenv("LLM_BASE_URL") or getattr(args, 'api_base', None)
    model_name = os.getenv("LLM_MODEL") or args.llm

    if not api_key:
        print("[FATAL] LLM_API_KEY not set in environment or args.")
        return ""

    for attempt in range(max_retries):
        try:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
                temperature=0.0,
                max_retries=5,
                timeout=120,
            )

            messages = [
                SystemMessage(content="You are an AI assistant that strictly follows instructions."),
                HumanMessage(content=prompt_text),
            ]

            response = llm.invoke(messages)
            return response.content

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[RETRY] LLM query attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("\n" + "="*50)
                print(f"[FATAL ERROR after {max_retries} attempts in _query_llm_with_langchain]: {e}")
                traceback.print_exc()
                print("="*50 + "\n")
                return ""

class NeighborsDataset(Dataset):
    def __init__(self, args, dataset, indices, query_index, pred, p, cluster_name=None, num_neighbors=None,
                 di_all=None, di_all_pos_cluster_idx=None, di_all_neg_cluster_idx=None):
        super(NeighborsDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.indices = indices
        # Convert query_index to a set of python ints for O(1) lookup and type safety
        self.query_index = set([int(i) for i in query_index])
        print(f"DEBUG: NeighborsDataset initialized with {len(self.query_index)} query indices")
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.pred = pred
        self.p = p
        self.cluster_name = cluster_name

        self.count = 0
        self.di = {}
        self.di_pos_cluster_idx = {}
        self.di_neg_cluster_idx = {}

        # Cache mechanism
        self.di_all = di_all if di_all is not None else {}
        self.di_all_pos_cluster_idx = di_all_pos_cluster_idx if di_all_pos_cluster_idx is not None else {}
        self.di_all_neg_cluster_idx = di_all_neg_cluster_idx if di_all_neg_cluster_idx is not None else {}

        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = list(self.dataset.__getitem__(index))
        neighbor_pred = np.take(self.pred, self.indices[index, :])

        # unique prediction
        res = [neighbor_pred[0]]
        for i in neighbor_pred[1:]:
            if i not in res:
                res.append(i)
                break


        pos_cluster_idx = None
        neg_cluster_idx = None
        # Ensure index is int for set lookup
        idx_key = int(index)
        if self.args.running_method not in ['Loop', 'GCD', 'SimGCD', 'BaCon']:
            ## Ours
            if idx_key not in self.query_index:
                if self.di_all.get(idx_key, -1) == -1:
                    # For the unselected samples, randomly select a sample from their neighbors
                    neighbor_index = np.random.choice(self.indices[idx_key], 1)[0]
                else:
                    # If they have been queried in previous rounds, use the LLM selected neighbors
                    neighbor_index = self.di_all[idx_key]
                    if self.args.weight_cluster_instance_cl > 0:
                        pos_cluster_idx = self.di_all_pos_cluster_idx[idx_key]
                        neg_cluster_idx = self.di_all_neg_cluster_idx[idx_key]

            else:
                # For the selected samples, query llm to select the most similar sample from the neighboring clusters
                anchor_text = self.tokenizer.decode(anchor[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                if self.di.get(idx_key, -1) == -1:
                    prob_tensor = self.p[idx_key, :]
                    topk_probs, topk_indices = torch.topk(prob_tensor, self.args.options, dim=-1)
                    qs = [np.random.choice(np.where(self.pred==topk_indices[i].item())[0], 1)[0] for i in range(self.args.options)]
                    # neighbor_index = self.query_llm_gen(index, qs)
                    neighbor_index, confidence = self.query_llm_gen(idx_key, qs)

                    if self.args.flag_filtering:
                        # filter out the LLM feedback with confidence less than a threshold
                        if float(confidence) < self.args.filter_threshold:
                            neighbor_index = np.random.choice(self.indices[idx_key], 1)[0]

                    self.di[idx_key] = neighbor_index
                    self.di_all[idx_key] = neighbor_index


                    if self.args.weight_cluster_instance_cl > 0:
                        # query llm to assign the anchor to one of the topk clusters based on category names and descriptions
                        k = int(np.floor(self.args.options_cluster_instance_ratio * len(self.cluster_name)))
                        topk_probs, topk_indices = torch.topk(prob_tensor, k, dim=-1)
                        topk_cluster_name = [self.cluster_name[i.item()] for i in topk_indices]
                        pos_cluster_idx, confidence = self.query_llm_cluster_instance(anchor_text, topk_cluster_name, topk_indices)
                        neg_cluster_idx = topk_indices[topk_indices != pos_cluster_idx]

                        if self.count < 6:
                            print(f"\nAnchor: {anchor_text} \nPositive Cluster Name: {self.cluster_name[pos_cluster_idx]}")

                        if self.args.flag_filtering_c:
                            # filter out the LLM feedback with confidence less than a threshold
                            if float(confidence) < self.args.filter_threshold_c:
                                pos_cluster_idx = None
                                neg_cluster_idx = None

                        self.di_all_pos_cluster_idx[idx_key] = pos_cluster_idx
                        self.di_all_neg_cluster_idx[idx_key] = neg_cluster_idx

                    self.di_pos_cluster_idx[idx_key] = pos_cluster_idx
                    self.di_neg_cluster_idx[idx_key] = neg_cluster_idx
                    self.count += 1

                else:
                    neighbor_index = self.di[idx_key]
                    if self.args.weight_cluster_instance_cl > 0:
                        pos_cluster_idx = self.di_pos_cluster_idx[idx_key]
                        neg_cluster_idx = self.di_neg_cluster_idx[idx_key]

        else:
            ## Generalized Loop
            # For the unselected samples, randomly select a sample from their neighbors
            if len(res) == 1 or idx_key not in self.query_index or self.args.running_method in ['GCD', 'SimGCD']:
                neighbor_index = np.random.choice(self.indices[idx_key], 1)[0]
            else:
                # For the selected samples, randomly select a sample from its top neighboring clusters
                # Generalize to the case # querying neighbors / options >= 2
                qs = [np.random.choice(self.indices[idx_key, np.where(neighbor_pred==res[i])][0], 1)[0] for i in range(self.args.options)]

                if self.di.get(idx_key, -1) == -1:
                    # Generalize to the case # querying neighbors / options >= 2
                    # neighbor_index = self.query_llm_gen(index, qs)
                    neighbor_index, confidence = self.query_llm_gen(idx_key, qs)

                    if self.args.flag_filtering:
                        # filter out the LLM feedback with confidence less than a threshold
                        if float(confidence) < self.args.filter_threshold:
                            neighbor_index = np.random.choice(self.indices[idx_key], 1)[0]
                            self.di[idx_key] = neighbor_index

                    self.di[idx_key] = neighbor_index
                    self.count += 1
                else:
                    neighbor_index = self.di[idx_key]


        neighbor = self.dataset.__getitem__(neighbor_index)
        output['anchor'] = anchor[:3]
        output['neighbor'] = neighbor[:3]
        output['possible_neighbors'] = torch.from_numpy(self.indices[index]) # used for neighbor contrastive learning
        output['target'] = anchor[-1]
        output['index'] = index
        output['pos_cluster_idx'] = pos_cluster_idx
        output['neg_cluster_idx'] = neg_cluster_idx

        return output


    def query_llm_gen(self, q_index, qs_indices):
        anchor_text = self.tokenizer.decode(self.dataset.__getitem__(q_index)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        candidate_texts = [self.tokenizer.decode(self.dataset.__getitem__(idx)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True) for idx in qs_indices]

        prompt = "Select the text that better corresponds with the Query in terms of topic or category. "
        prompt += "\n Also show your confidence by providing a probability between 0 and 1."
        prompt += "\n Please respond in the format 'Choice [number], Confidence: [number]' without explanation, e.g., 'Choice 1, Confidence: 0.7'.\n"

        if getattr(self.args, 'flag_demo', False):
            prompt += self.args.prompt_demo

        prompt += f"\nQuery: {anchor_text}"
        for i, text in enumerate(candidate_texts):
            prompt += f"\nChoice {i + 1}: {text}"

        if self.count < 5: print(f"\n--- Neighbor Selection Prompt ---\n{prompt}\n------------------------------")

        response_content = _query_llm_with_langchain(prompt, self.args)

        # DEBUG: Print raw response for verification
        print(f"\n[DEBUG LLM Raw Response]: {response_content}")

        try:
            content = response_content.lower()
            confidence_match = re.search(r'(?:confidence):\s*(\d\.?\d*)', content)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0

            choice_idx = -1
            for i in range(len(qs_indices)):
                choice_str = f'choice {i + 1}'
                if choice_str in content:
                    choice_idx = i
                    print(f"[DEBUG LLM Parsed]: Successfully matched Choice {i+1}, Confidence: {confidence}")
                    return qs_indices[i], confidence

            # Additional fallback for pure number response if only one candidate requested (though qs_indices length depends on options)
            if len(qs_indices) == 2:
                if '1' == content.strip():
                    print(f"[DEBUG LLM Parsed]: Fallback matched Choice 1 (pure number), Confidence: {confidence}")
                    return qs_indices[0], confidence
                if '2' == content.strip():
                    print(f"[DEBUG LLM Parsed]: Fallback matched Choice 2 (pure number), Confidence: {confidence}")
                    return qs_indices[1], confidence

            print(f"[DEBUG LLM Parsed]: FAILED to match any choice. Defaulting to Choice 1, Confidence: 0.0")
            return qs_indices[0], 0.0
        except Exception:
            return qs_indices[0], 0.0

    def query_llm_cluster_instance(self, anchor_text, topk_cluster_name, topk_cat_indices):
        prompt = "Select the category that better corresponds with the Query in terms of topic or category. "
        prompt += "\n Also show your confidence by providing a probability between 0 and 1."
        prompt += "\n Please respond in the format 'Choice [number], Confidence: [number]' without explanation, e.g., 'Choice 1, Confidence: 0.7'.\n"

        if getattr(self.args, 'flag_demo_c', False):
            prompt += self.args.prompt_demo_c

        prompt += f"\nQuery: {anchor_text}"
        for i, cluster_name in enumerate(topk_cluster_name):
            prompt += f"\nChoice {i + 1}: {cluster_name}"

        if self.count < 5: print(f"\n--- Cluster Alignment Prompt ---\n{prompt}\n----------------------------")

        response_content = _query_llm_with_langchain(prompt, self.args)

        # DEBUG: Print raw response for verification
        print(f"\n[DEBUG Cluster Raw Response]: {response_content}")

        try:
            content = response_content.lower()
            confidence_match = re.search(r'(?:confidence):\s*(\d\.?\d*)', content)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0

            for i in range(len(topk_cat_indices)):
                choice_str = f'choice {i + 1}'
                if choice_str in content:
                    print(f"[DEBUG Cluster Parsed]: Successfully matched Choice {i+1}, Confidence: {confidence}")
                    return topk_cat_indices[i].item(), confidence

            print(f"[DEBUG Cluster Parsed]: FAILED to match any choice. Defaulting to Choice 1, Confidence: 0.0")
            return topk_cat_indices[0].item(), 0.0
        except Exception:
            return topk_cat_indices[0].item(), 0.0
