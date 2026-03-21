################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

from datasets import load_dataset
from termcolor import colored
import random
import numpy as np
import json
import os
from math import inf

# RULER
from .metrics import needle_score, string_match_part, multi_number, multi_words

# LongBench, InfiniBench
from .metrics import long_bench_metrics, get_infinibench_scorer

# NIAH
from data.utils import generate_random_number, read_context_files, create_contexts, NIAH_TEMPLATE, RANDOM_NEEDLE_CITIES

METRICS_FN = {
    'niah': needle_score,
    'multi': multi_number,
    'vt': multi_words,
    'cwe': multi_words,
    'fwe': multi_words,
    'qa': string_match_part,
}

GEN_LEN = {
    'niah': 64,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32,
}

DATADIR = {
    'ruler': 'data/ruler/data',
    'niah': 'data/niah/data',
}

class Dataset:
    def __init__(self, dataset_name, tokenizer, datalen, num_samples, rank=0, world_size=1):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.datalen = datalen
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.is_sharded = False

        if dataset_name == 'niah':
            self.tokenized_prompts, self.gt, self.ctx_len, self.depth_pct = self.get_dataset()
        elif 'long_bench' in dataset_name:
            self.tokenized_prompts, self.gt, self.classes = self.get_dataset()
        else:
            self.tokenized_prompts, self.gt = self.get_dataset()
        
        self.num_samples = len(self.gt)
        self.gen_len = self.get_gen_len()
        self.metric = self.get_metric()

    def __str__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __repr__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __len__(self) -> int:
        return self.num_samples

    def shard(self, rank, world_size):
        if world_size > 1:
            shard_size = self.num_samples // world_size
            start = rank * shard_size
            end = start + shard_size if rank != world_size - 1 else self.num_samples
            shard_tokenized_prompts, shard_gt = self.tokenized_prompts[start:end], self.gt[start:end]
            if self.classes is not None:
                self.classes = self.classes[start:end]
            if self.prompts is not None:
                self.prompts = self.prompts[start:end]
            self.tokenized_prompts = shard_tokenized_prompts
            self.gt = shard_gt
            self.num_samples = len(shard_tokenized_prompts)

        self.is_sharded = True

    def get_gen_len(self):
        if 'long_bench' in self.dataset_name:
            task_name = self.dataset_name.split('/')[-1]
            # Load gen_len from config file
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'long_bench/config/dataset2maxlen.json')
            with open(config_path, 'r') as f:
                gen_len_config = json.load(f)
            if task_name in gen_len_config:
                return gen_len_config[task_name]
            else:
                raise Exception(f"Maxlen not found from config for LongBench task: {task_name}")
        elif 'niah' ==  self.dataset_name:
            return 10
        elif 'niah' in self.dataset_name:
            return 128
        elif 'vt' in self.dataset_name:
            return 30
        elif 'cwe' in self.dataset_name:
            return 120
        elif 'fwe' in self.dataset_name:
            return 50
        elif 'qa' in self.dataset_name:
            return 32
        elif 'infini_bench' in self.dataset_name:
            if task_name == 'longbook_sum_eng':
                return 2048
            return 32
        else:
            raise Exception("Gen len not found")

    def __getitem__(self, idx):
        if 'persona' in self.dataset_name:
            return self.tokenized_prompts[idx], self.queries[idx], self.gt[idx]
        elif 'long_bench' in self.dataset_name:
            return self.tokenized_prompts[idx], self.gt[idx], self.classes[idx]
        return self.tokenized_prompts[idx], self.gt[idx]

    # NOTE: get scorer according to dataset_name
    def get_metric(self):
        if 'long_bench' in self.dataset_name:
            task_name = self.dataset_name.split('/')[-1]
            return long_bench_metrics[task_name]
        elif 'multiquery' in self.dataset_name or 'multivalue' in self.dataset_name:
            return METRICS_FN['multi']
        elif 'niah' in self.dataset_name:
            return METRICS_FN['niah']
        elif 'vt' in self.dataset_name:
            return METRICS_FN['vt']
        elif 'cwe' in self.dataset_name:
            return METRICS_FN['cwe']
        elif 'fwe' in self.dataset_name:
            return METRICS_FN['fwe']
        elif 'qa' in self.dataset_name:
            return METRICS_FN['qa']
        elif 'infini_bench' in self.dataset_name:
            task_name = self.dataset_name.split('/')[-1]
            return get_infinibench_scorer(task_name)
        else:
            raise Exception("Metric not found")

    # NOTE: jFetch and process dataset
    def get_dataset(self):
        if 'ruler' in self.dataset_name: # ruler/xxx
            task = self.dataset_name.split('/')[-1]
            assert self.datalen in [8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024], "Only support datalen of 16k, 32k, 64k, 128k"

            if 'llama-3' in self.tokenizer.name_or_path.lower():
                model_dir = 'llama-3'
            elif 'yi' in self.tokenizer.name_or_path.lower():
                model_dir = 'yi'
            elif 'lwm' in self.tokenizer.name_or_path.lower():
                model_dir = 'lwm'
            elif 'glm' in self.tokenizer.name_or_path.lower():
                model_dir = 'glm'
            elif 'qwen' in self.tokenizer.name_or_path.lower():
                model_dir = 'qwen'
            elif 'phi' in self.tokenizer.name_or_path.lower():
                model_dir = 'phi'
            else:
                raise Exception("Model not found", self.tokenizer.name_or_path)

            dataset = load_dataset("json", data_files=f'{DATADIR["ruler"]}/{model_dir}/{self.datalen}/{task}/validation.jsonl', split='train')
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            gts = []

            for i in range(self.num_samples):
                input_text = dataset[i]['input']
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
                tokenized_prompts.append(input_ids)
                gts.append(dataset[i]['outputs'])

            return tokenized_prompts, gts

        elif self.dataset_name == 'niah':
            print(colored(f"[Warning] NIAH dataset cannot set # samples, it is up to world_size, which is set to {self.world_size}", 'red'))
            
            haystack_file = f'{DATADIR["niah"]}/pg19_mini.jsonl'
            context_lengths_min = 16*1024
            context_lengths_max = self.datalen
            n_context_length_intervals = 15
            n_document_depth_intervals = 10  # position of the needle in the haystack
            n_rounds = 1 # max(1, 4 // self.world_size) # 8 rounds in total assume we have 8xGPUs
            needle = "\nThe special magic {city} number is: {rnd_number}\n"
            retrieval_question="What is the special magic {} number?"
            rnd_number_digits = 7

            context_lengths = np.round(
                np.linspace(
                    context_lengths_min,
                    context_lengths_max,
                    num=n_context_length_intervals,
                    endpoint=True,
                )
            ).astype(int)

            document_depth_percents = np.round( # we use linear scale here
                np.linspace(
                    0,
                    100,
                    num=n_document_depth_intervals,
                    endpoint=True,
                )
            ).astype(int)

            self.is_sharded = True # we shard the data during init dataset
            
            full_contexts = read_context_files(n=n_rounds, context_lengths=context_lengths, haystack_file=haystack_file, tokenizer=self.tokenizer)
            full_tokens = [
                self.tokenizer.encode(full_context, add_special_tokens=False) for full_context in full_contexts
            ]

            tokenized_prompts = []
            gts = []
            ctx_len = []
            depth_pct = []

            for context_length in context_lengths:
                trim_contexts = [
                    self.tokenizer.decode(full_token[:context_length], skip_special_tokens=True)
                    for full_token in full_tokens
                ]
                contexts = []
                for depth_percent in document_depth_percents:
                    for i in range(n_rounds):
                        random_city = random.choice(RANDOM_NEEDLE_CITIES)
                        insert_needle = True
                        needle_rnd_number = str(generate_random_number(rnd_number_digits))
                        context = create_contexts(
                            needle_rnd_number=needle_rnd_number,
                            insert_needle=insert_needle,
                            random_city=random_city,
                            trim_context=trim_contexts[i],
                            context_length=context_length,
                            depth_percent=depth_percent,
                            needle=needle,
                            retrieval_question=retrieval_question,
                            tokenizer=self.tokenizer,
                            final_context_length_buffer=32,
                        )
                        contexts.append(context)

                for context in contexts:
                    prompt = NIAH_TEMPLATE.format(
                        context=context["context"], question=context["question"]
                    )
                    input_tensor = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
                    tokenized_prompts.append(input_tensor.input_ids)
                    gts.append(context["needle_rnd_number"])
                    ctx_len.append(context["context_length"])
                    depth_pct.append(context["depth_percent"])
            
            return tokenized_prompts, gts, ctx_len, depth_pct

        elif 'long_bench' in self.dataset_name:
            # Extract task name from dataset_name (e.g., 'long_bench/narrativeqa' -> 'narrativeqa')
            task_name = self.dataset_name.split('/')[-1]
            
            print(colored(f"Loading LongBench task: {task_name}", 'cyan'))
            
            # Load task template from config file
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'long_bench/config/dataset2prompt.json')
            with open(config_path, 'r') as f:
                dataset2prompt = json.load(f)
            if task_name not in dataset2prompt:
                raise ValueError(f"Prompt template not found for LongBench task: {task_name}")
            task_template = dataset2prompt[task_name]
            
            # Load dataset from HuggingFace with local caching
            dataset = load_dataset(
                'THUDM/LongBench', 
                task_name,
                split='test',
                trust_remote_code=True,
            )
            
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            
            tokenized_prompts = []
            gts = []
            all_classes_list = []

            min_length = 4 * 1024 # minimal allowed length of prompt
            
            # truncate datalen
            trunc_cnt = 0
            trunc_len = 0
            filter_cnt = 0  # Count filtered samples

            sample_max_tokens = 0
            sample_min_tokens = inf

            for i in range(self.num_samples):
                sample = dataset[i]
                
                # Format prompt using loaded template
                prompt = task_template.format(**sample)
                
                # Filter samples with context length < 4k tokens
                tokenized_prompt = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
                if tokenized_prompt.shape[-1] < min_length:
                    filter_cnt += 1
                    continue
                
                # Truncate prompt to fit size limited datalen
                if tokenized_prompt.shape[-1] > self.datalen:
                    half = self.datalen // 2
                    prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                    trunc_cnt += 1
                    trunc_len += tokenized_prompt.shape[-1] - self.datalen
                    tokenized_prompt = self.tokenizer.encode(prompt, return_tensors='pt')
                tokenized_prompts.append(tokenized_prompt)
                sample_max_tokens = max(sample_max_tokens, len(tokenized_prompt))
                sample_min_tokens = min(sample_min_tokens, len(tokenized_prompt))
                
                # Extract answers (gt as list for multi-answer support)
                answers = sample['answers'] if isinstance(sample['answers'], list) else [sample['answers']]
                gts.append(answers)
                
                # Save all_classes if available (needed for classification tasks)
                if 'all_classes' in sample:
                    all_classes_list.append(sample['all_classes'])
                else:
                    all_classes_list.append(None)
            
            print(f"Filtered samples with < 4k tokens: {filter_cnt}")
            print(f"Truncated Prompt Count: {trunc_cnt}, Truncated Prompt Avg Length: {trunc_len / trunc_cnt if trunc_cnt > 0 else 0}")
            print(colored(f"Loaded {len(tokenized_prompts)} examples for LongBench task '{task_name}'", 'green'))
            print(colored(f"max tokens: {sample_max_tokens}, min tokens: '{sample_min_tokens}'", 'green'))
            return tokenized_prompts, gts, all_classes_list

        elif 'infini_bench' in self.dataset_name: # infini_bench/xxx
            from data.infinibench.eval import VANILLA_INFINI_BENCH_TEMPLATE, infini_bench_create_prompt, infini_bench_get_answer, truncate_by_tokens
            
            task = self.dataset_name.split('/')[-1]
            dataset = load_dataset("xinrongzhang2022/InfiniteBench", split=task, num_proc=16)
            ### {'id':xxx, 'context': xxx, 'input': xxx, 'answer':xxx}
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            gt = []

            for i in range(len(dataset)):
                if 'llama-3' in self.tokenizer.name_or_path.lower():
                    model_template = VANILLA_INFINI_BENCH_TEMPLATE[task]
                elif 'yi' in self.tokenizer.name_or_path.lower():
                    model_template = VANILLA_INFINI_BENCH_TEMPLATE[task]
                elif 'glm' in self.tokenizer.name_or_path.lower():
                    model_template = VANILLA_INFINI_BENCH_TEMPLATE[task]
                else:
                    raise Exception("Model not found", self.tokenizer.name_or_path)

                input_text = infini_bench_create_prompt(dataset[i], task, model_template)
                input_ids = truncate_by_tokens(input_text, self.tokenizer, self.datalen)

                if input_ids.shape[-1] <= self.datalen:
                    tokenized_prompts.append(input_ids)
                    gt.append(infini_bench_get_answer(dataset[i], task))
                
                if len(tokenized_prompts) == self.num_samples:
                    break

            return tokenized_prompts, gt

        else:
            raise ValueError(f"Dataset {self.dataset_name} not found, please choose in ruler, persona, infini_bench, needle, niah, long_bench")