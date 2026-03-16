import re
import torch


VANILLA_INFINI_BENCH_TEMPLATE = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",
    
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",
    
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below. A specified key value pair is hidden within the following text. Make sure to memorize it. I will quiz you about the key value pair afterwards.\n\n{context}\n\nWhat is the specified value for '{input}' mentioned in the provided JSON? Please do not reply with the key, but with the value corresponding to the key.The value associated with '{input}' is:",

    "longbook_sum_eng": "Summarize the book below. \n\n{context}\n\nSummary:",

    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",
    
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",
    
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n请用中文回答。\n问题：{question}\n答案：",
    
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    
    "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",
    
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",
    
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\n{input} Just give the name without other words. Do not give me random numbers or something else. The name that has been replaced with \"$$MASK$$\" is ",
}

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