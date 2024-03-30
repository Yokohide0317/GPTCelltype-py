import os
import numpy as np
import pandas as pd
from openai import OpenAI

def gpt_celltype(_input, tissuename=None, model='gpt-4', topgenenumber=10):
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Note: OpenAI API key not found: returning the prompt itself.")
        api_flag = False
    else:
        print("Note: OpenAI API key found: proceeding with cell type annotations.")
        api_flag = True
        client = OpenAI(api_key=openai_api_key)

    if isinstance(_input, list):
        input_str = [','.join(map(str, sublist)) for sublist in _input]
    elif isinstance(_input, pd.DataFrame):
        _input = _input.groupby('group')['names'].apply(lambda x: ','.join(x.iloc[:topgenenumber])).to_dict()
    else:
        raise ValueError("Input must be a list or pandas DataFrame.")

    if not api_flag:
        message = f"Identify cell types of {tissuename} cells using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n" + "\n".join([f"{k}: {v}" for k, v in _input.items()])
        return message
    else:
        cutnum = np.ceil(len(_input) / 30)
        all_results = {}

        for i in range(1, int(cutnum) + 1):
            ids = [k for j, k in enumerate(_input.keys()) if np.ceil((j + 1) / 30) == i]
            flag = False

            while not flag:
                prompt = f"Identify cell types of {tissuename} cells using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n" + "\n".join([f"{k}: {_input[k]}" for k in ids])
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                )
                res = response.choices[0].message.content.split('\n')
                if len(res) == len(ids):
                    flag = True
            all_results.update(dict(zip(ids, res)))

        print("Note: It is always recommended to check the results returned by GPT-4 in case of AI hallucination, before going to down-stream analysis.")
        return all_results