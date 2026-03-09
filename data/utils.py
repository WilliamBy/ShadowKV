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
# Some code comes from experments/ in MInference
# Original license:
# Copyright (c) Microsoft Corporation. and affiliates All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

import re
import json
import random
import torch

def truncate_input(input: torch.LongTensor, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if input.shape[-1] <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return torch.cat([input[:, 0:split],input[:, -split:]], dim=-1)
    else:
        return None

def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input, return_tensors="pt")
    len_before = tokens.shape[-1]
    # print(f"# tokens before: {len_before} (max_tokens: {max_tokens})")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = tokens.shape[-1]  # type: ignore
    # print(f"# tokens after: {len_after} (max_tokens: {max_tokens})")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens

########## NIAH ##########

NIAH_TEMPLATE = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n{context}\n\nQuestion: {question} Don't give information outside the document or repeat your findings. Keep your response short and direct. Answer: "

RANDOM_NEEDLE_CITIES = ["Chicago", "Yangon", "Antananarivo", "Colombo", "Almaty", "Sydney", "Chicago", "Mexico City", "Seattle", "Lagos", "Amsterdam", "Belgrade", "Cairo", "Baghdad", "Damascus", "Kigali", "Dakar", "Dakar", "Sofia", "Kigali", "Victoria", "Tashkent", "Mumbai", "Barcelona", "Almaty", "Amman", "Toronto", "Bratislava", "Johannesburg", "Thimphu", "Bangkok", "Santiago", "Cairo", "San Francisco", "Lagos", "Amsterdam", "Paris", "Rabat", "Santiago", "Copenhagen", "Madrid", "Kigali", "Ho Chi Minh City", "Sarajevo", "Delhi", "Istanbul", "Ho Chi Minh City", "Khartoum", "Helsinki", "Doha", "Istanbul", "Kuala Lumpur", "Budapest", "Shanghai", "Moscow", "Los Angeles", "Oslo", "Johannesburg", "Berlin", "Bangalore", "Tokyo", "Melbourne", "Barcelona", "Chicago", "Port Louis", "Lisbon", "Nairobi", "Kampala", "Lima", "Maputo", "Vancouver", "Dubai", "Khartoum", "Jakarta", "Madrid", "Yerevan", "Beirut", "Athens", "Chicago", "Paris", "Bucharest", "Copenhagen", "Brussels", "Damascus", "Seattle", "Los Angeles", "Yerevan", "Victoria", "Tunis", "Astana", "Seoul", "Buenos Aires", "Bangkok", "Colombo", "Brussels", "Khartoum", "Doha", "San Francisco", "Vienna", "Jakarta"]

def generate_random_number(num_digits):
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return random.randint(lower_bound, upper_bound)

def read_context_files(n, context_lengths, haystack_file, tokenizer):
    max_context_length = max(context_lengths)
    contexts = []
    f = open(haystack_file, "r")
    for _ in range(n):
        context = ""
        toks = 0
        while toks < max_context_length:
            text = json.loads(f.readline())["text"]
            context += text
            toks += len(tokenizer.encode(text))
        contexts.append(context)
    return contexts

def insert_needle_func(needle, context, depth_percent, context_length, tokenizer, final_context_length_buffer):
    tokens_needle = tokenizer.encode(needle, add_special_tokens=False)
    tokens_context = tokenizer.encode(context, add_special_tokens=False)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= final_context_length_buffer

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[: context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = [tokenizer.encode(".", add_special_tokens=False)[0], tokenizer.encode(". \n", add_special_tokens=False)[0], tokenizer.encode(".\n", add_special_tokens=False)[0], tokenizer.encode("\n", add_special_tokens=False)[0]]

        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    # Convert back to a string and return it
    new_context = tokenizer.decode(tokens_new_context, skip_special_tokens=True)
    return new_context

def create_contexts(
    needle_rnd_number,
    insert_needle,
    random_city,
    trim_context,
    context_length,
    depth_percent,
    needle,
    retrieval_question,
    tokenizer,
    final_context_length_buffer,
):
    needle = needle.format(city=random_city, rnd_number=needle_rnd_number)
    question = retrieval_question.format(random_city)
    if not insert_needle:
        needle = " "  # replace needle with a space
    context = insert_needle_func(
        needle, trim_context, depth_percent, context_length, tokenizer, final_context_length_buffer
    )
    results = {
        "context": context,
        "context_length": int(context_length),
        "depth_percent": float(depth_percent),
        "needle": needle,
        "question": question,
        "insert_needle": insert_needle,
        "needle_rnd_number": needle_rnd_number,
    }
    return results

def infini_bench_create_prompt(eg: dict, data_name: str, template: str) -> str:
    # Code tasks
    if data_name == "code_run":
        find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
        func_call = find_result[0]
        func = func_call.split("(")[0]
        return template.format(
            func=func,
            func_call=func_call,
            context=eg["context"],
        )
    elif data_name in ["code_debug", "code_debug_qa"]:
        code = eg["context"]
        if data_name == "code_debug":
            return template.format(
                context=code,
                OPTION_A=eg["options"][0],
                OPTION_B=eg["options"][1],
                OPTION_C=eg["options"][2],
                OPTION_D=eg["options"][3],
            )
        return template.format(
            context=code,
        )
    # Long book tasks
    elif data_name in [
        "longbook_choice_eng",
        "longbook_qa_eng",
        "longbook_sum_eng",
        "longbook_qa_chn",
    ]:
        book = eg["context"]
        if data_name == "longbook_choice_eng":
            return template.format(
                question=eg["input"],
                context=book,
                OPTION_A=eg["options"][0],
                OPTION_B=eg["options"][1],
                OPTION_C=eg["options"][2],
                OPTION_D=eg["options"][3],
            )
        elif data_name == "longbook_qa_eng":
            return template.format(
                question=eg["input"],
                context=book,
            )
        elif data_name == "longbook_sum_eng":
            return template.format(
                context=book,
            )
        elif data_name == "longbook_qa_chn":
            return template.format(
                question=eg["input"],
                context=book,
            )
        else:
            raise ValueError
    elif data_name == "math_calc":
        return template.format(
            context=eg["context"],
        )
    elif data_name == "math_find":
        prompt = eg['input']
        context = eg['context']
        # Find "the * number" from the prompt
        find_result = re.findall(r"The .+ of", prompt)
        assert find_result, f"Cannot find the target number in {prompt}"
        target_number = find_result[0].lower()[:-3]
        # Replace the number with the answer
        prefix = f"What is {target_number} in the following list?"
        return template.format(
            prefix=prefix,
            context=context,
            input=prompt,
        )

    if "content" in eg:
        content = eg["content"]
        del eg["content"]
        eg["context"] = content

    format_dict = {
        "context": eg["context"],
        "input": eg["input"],
    }

    if data_name == "kv_retrieval":
        format_dict["input"] = eg["input"].split('"')[1]

    prompt = template.format(**format_dict)
    return prompt

def infini_bench_get_answer(eg: dict, data_name: str):
    if data_name in ["code_debug", "longbook_choice_eng"]:
        OPTIONS = "ABCD"
        if isinstance(eg["answer"], str):
            ret = [eg["answer"], OPTIONS[eg['options'].index(eg["answer"])]]
        elif isinstance(eg["answer"], list):
            if len(eg["answer"]) == 1:
                ret = [eg["answer"][0], OPTIONS[eg['options'].index(eg["answer"][0])]]
            elif len(eg["answer"]) == 2 and eg["answer"][1] in ['A', 'B', 'C', 'D']:
                ret = eg['answer']
            else:
                raise ValueError
        else:
            raise ValueError
        return ret

    return eg["answer"]