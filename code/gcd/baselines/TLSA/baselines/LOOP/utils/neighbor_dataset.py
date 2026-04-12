import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from heapq import nlargest
import time
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from utils.token_tracker import update_token_usage

class NeighborsDataset(Dataset):
    def __init__(self, args, dataset, indices, query_index, pred, num_neighbors=None, llm_cache=None):
        super(NeighborsDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        # Convert query_index to a set for O(1) lookup
        self.query_index = set([int(i) for i in query_index])
        print(f"DEBUG: NeighborsDataset initialized with {len(self.query_index)} query indices")
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.pred = pred
        self.count = 0
        self.di = llm_cache if llm_cache is not None else {}
        assert(self.indices.shape[0] == len(self.dataset))

        # Persistent LLM client
        api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or getattr(self.args, 'api_key', '')
        base_url = os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_API_BASE') or getattr(self.args, 'api_base', '')
        self.llm_model = os.getenv('LLM_MODEL') or getattr(self.args, 'llm_model_name', '') or 'gpt-3.5-turbo'

        if api_key and api_key != 'EMPTY':
            self.llm = ChatOpenAI(
                model=self.llm_model,
                base_url=base_url if base_url else None,
                api_key=api_key,
                temperature=0.0,
                max_tokens=256,
                timeout=240,
            )
        else:
            self.llm = None

        if self.llm:
            self.pre_query_llm_parallel()

    def pre_query_llm_parallel(self):
        """Parallelly pre-query all samples in query_index that are not in cache."""
        to_query = []
        for idx in self.query_index:
            if idx not in self.di:
                neighbor_pred = np.take(self.pred, self.indices[idx, :])
                res = [neighbor_pred[0]]
                for i in neighbor_pred[1:]:
                    if i not in res:
                        res.append(i)
                        break

                if len(res) > 1:
                    q1 = np.random.choice(self.indices[idx, np.where(neighbor_pred==res[0])][0], 1)[0]
                    q2 = np.random.choice(self.indices[idx, np.where(neighbor_pred==res[1])][0], 1)[0]
                    to_query.append((idx, q1, q2))

        if not to_query:
            return

        print(f"DEBUG: Pre-querying {len(to_query)} samples with LLM in parallel...")
        max_workers = min(len(to_query), 20) # Adjust parallel workers as needed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.query_llm, q, q1, q2): q for q, q1, q2 in to_query}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    neighbor_index = future.result()
                    self.di[idx] = neighbor_index
                except Exception as e:
                    print(f"DEBUG: Error in parallel pre-query for index {idx}: {e}")
                    # Fallback to q1 if query failed (handled inside query_llm anyway)
        print(f"DEBUG: Parallel pre-querying finished. Cache size: {len(self.di)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = list(self.dataset.__getitem__(index))
        neighbor_pred = np.take(self.pred, self.indices[index, :])

        res = [neighbor_pred[0]]
        for i in neighbor_pred[1:]:
            if i not in res:
                res.append(i)
                break

        # Use int(index) to ensure lookup in set works correctly
        idx_key = int(index)
        if len(res) == 1 or idx_key not in self.query_index:
            all_neighbors = self.indices[index]
            valid_neighbors = self._get_candidates(all_neighbors, index)
            neighbor_index = np.random.choice(valid_neighbors, 1)[0]
        else:
            if idx_key in self.di:
                neighbor_index = self.di[idx_key]
            else:
                # If for some reason it's not pre-queried (should not happen with pre_query_llm_parallel)
                cluster1_indices = self.indices[index, np.where(neighbor_pred==res[0])][0]
                valid_q1 = self._get_candidates(cluster1_indices, index)
                q1 = np.random.choice(valid_q1, 1)[0]

                cluster2_indices = self.indices[index, np.where(neighbor_pred==res[1])][0]
                valid_q2 = self._get_candidates(cluster2_indices, index)
                q2 = np.random.choice(valid_q2, 1)[0]

                neighbor_index = self.query_llm(idx_key, q1, q2)
                self.di[idx_key] = neighbor_index

        neighbor = self.dataset.__getitem__(neighbor_index)
        output['anchor'] = anchor[:3]
        output['neighbor'] = neighbor[:3]
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor[-1]
        output['index'] = index
        return output

    def query_llm(self, q, q1, q2):
        s = self.tokenizer.decode(self.dataset.__getitem__(q)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        s1 = self.tokenizer.decode(self.dataset.__getitem__(q1)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        s2 = self.tokenizer.decode(self.dataset.__getitem__(q2)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if self.args.dataset == 'thucnews':
            prompt = "Select the news title/content that better corresponds with the Query in terms of topic/category. The data is Chinese news. Please respond with 'Choice 1' or 'Choice 2' without explanation. \n Query: " + s + "\n Choice 1: " + s1 + "\n Choice 2: " + s2
        else:
            prompt = "Select the utterance or text that better corresponds with the Query in terms of intent/topic. Please respond with 'Choice 1' or 'Choice 2' without explanation. \n Query: " + s + "\n Choice 1: " + s1 + "\n Choice 2: " + s2

        self.count += 1

        if not self.llm:
            return q1

        max_retries = 3
        retry_delay = 2 # seconds

        for attempt in range(max_retries):
            try:
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]

                response = self.llm.invoke(messages)
                update_token_usage(response)
                content = response.content.lower()

                # Parse response for choice selection
                if 'choice 1' in content or '1' == content.strip():
                    return q1
                elif 'choice 2' in content or '2' == content.strip():
                    return q2
                else:
                    return q1
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Exponential backoff
                else:
                    print(f"LLM query failed after {max_retries} attempts: {e}")
                    return q1
