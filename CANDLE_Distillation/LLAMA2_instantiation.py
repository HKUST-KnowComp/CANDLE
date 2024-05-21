import argparse
import json

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import trange
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=500000)
args = parser.parse_args()

start_index = args.start_index
end_index = args.end_index

if __name__ == '__main__':
    model = "meta-llama/Llama-2-13b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model, token="YOUR_HUGGINGFACE_TOKEN")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float32,
        device_map="auto",
        token="YOUR_HUGGINGFACE_TOKEN",
    )

    relation2text_dict = {
        'xEffect': ', as a result, PersonX will',
        'xReact': ', as a result, PersonX feels',
        'xWant': ', as a result, PersonX wants',
        'oEffect': ', as a result, PersonY or others will',
        'oReact': ', as a result, PersonY or others feels',
        'oWant': ', as a result, PersonY or others wants',
        'xIntent': ', because PersonX wanted to',
        'xAttr': ', in this way, PersonX is seen as',
        'xNeed': ', but before, PersonX needs to',
    }

    AbstractATOMIC = pd.read_csv('../../data/abstractATOMIC/triple_annotated.csv', index_col=None)
    AbstractATOMIC = AbstractATOMIC[AbstractATOMIC.label == 1].reset_index(drop=True)
    CANDLE_C = pd.read_csv('../../data/LLM_conceptualization/LLM_all_conceptualization.csv', index_col=None)

    if end_index > start_index:
        CANDLE_C = CANDLE_C[start_index:end_index].reset_index(drop=True)
    else:
        CANDLE_C = CANDLE_C[start_index:].reset_index(drop=True)

    AbstractATOMIC = AbstractATOMIC.sample(n=10).reset_index(drop=True)
    prompt = "Given the sentence: '{}{} {}', replacing the concept '{}' with an instance '{}' can make it more detailed: '{}{} {}'."
    example_prompt = ""
    for i in range(len(AbstractATOMIC)):
        filled_example = prompt.format(AbstractATOMIC.loc[i, 'head'].replace('[', '').replace(']', ''),
                                       relation2text_dict[AbstractATOMIC.loc[i, 'relation']],
                                       AbstractATOMIC.loc[i, 'tail'],
                                       AbstractATOMIC.loc[i, 'head'].split(']')[0].split('[')[-1],
                                       json.loads(AbstractATOMIC.loc[i, 'info'])['sent'].split('[')[-1].split(']')[0],
                                       json.loads(AbstractATOMIC.loc[i, 'info'])['sent'].replace('[', '').replace(']',
                                                                                                                  ''),
                                       relation2text_dict[AbstractATOMIC.loc[i, 'relation']],
                                       AbstractATOMIC.loc[i, 'tail'])
        example_prompt += filled_example + '\n\n'
    print(example_prompt)

    instruction = "Following the examples, complete the last sentence by replacing the concept with an instance. The generated sentence must start with PersonX or PersonY.\n\n"

    merged_sents = []
    for i in trange(len(CANDLE_C)):
        query = "Given the sentence: '{}{} {}', replacing the concept '{}' with an instance".format(
            CANDLE_C.loc[i, 'head'].replace(str(CANDLE_C.loc[i, 'instance']), str(CANDLE_C.loc[i, 'concept'])),
            relation2text_dict[CANDLE_C.loc[i, 'relation']], CANDLE_C.loc[i, 'tail'], CANDLE_C.loc[i, 'concept'])

        merged_sent = instruction + example_prompt + query

        print(len(tokenizer(merged_sent)['input_ids']))

        sequences = pipeline(
            merged_sent,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=250,
        )
        merged_sents.append(sequences)

        if i % 2000 == 0:
            np.save("./llama13b-chat_instantiations_{}_{}.npy".format(start_index, end_index), merged_sents)
