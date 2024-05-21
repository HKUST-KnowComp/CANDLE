import openai
import time
import asyncio
import json
import math
import pandas as pd
from tqdm import tqdm
from itertools import chain
import os

openai.api_key = "YOUR_OPENAI_KEY"


async def dispatch_openai_requests(
        messages_list,
        model="gpt-3.5-turbo-0301",
        max_tokens=256,
):
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
        )
        for x in messages_list
    ]
    time.sleep(1)
    return await asyncio.gather(*async_responses)


df = pd.read_csv("./v4_atomic_all_agg_instance_added.csv")
heads = df["event"]

all_instances = df['instance'].apply(lambda x: json.loads(str(x)) if isinstance(x, str) or not math.isnan(x) else [])

heads = df["event"]
templates_relation = {
    "xWant": "as a result, PersonX wants to",
    "oWant": "as a result, PersonY or others want to",
    "xEffect": "as a result, PersonX will",
    "oEffect": "as a result, PersonY or others will",
    "xReact": "as a result, PersonX feels",
    "oReact": "as a result, PersonY or others feel",
    "xAttr": "PersonX is seen as",
    "xIntent": "because PersonX wanted",
    "xNeed": "but before, PersonX needed",
}


def get_assertion(h, r, t):
    if t.startswith("to ") or t.startswith("To "):
        return f"If {h}, {templates_relation[r]} {t[3:]}."
    else:
        return f"If {h}, {templates_relation[r]} {t}."


r = "xWant"
all_tails = df[r].apply(lambda x: json.loads(str(x)) if isinstance(x, str) or not math.isnan(x) else [])
for tails in all_tails:
    while "none" in tails:
        tails.remove("none")

selected_event_instances_tails = [(heads[i], all_instances[i], all_tails[i]) for i in range(len(all_instances)) if
                                  len(all_instances[i]) > 0]
print(len(selected_event_instances_tails))

prompt_template = "Name 3-10 highly-confident noun phrases that can summarize the component \"{}\" from the assertion \"{}\"?"

bs = 1  # @param
total = (len(selected_event_instances_tails) - 1) // bs + 1
timeout_succeed = 3  # @param
timeout_failure = 10  # @param

out_filename = f'output/{r}_concept_0301_13.5k_total.txt'
if os.path.exists(out_filename):
    assert False, "outfile already exists!"

predictions = []
for i in tqdm(range(total)):
    while True:
        try:
            items = selected_event_instances_tails[i * bs:(i + 1) * bs]
            heads = list(chain(*[[item[0]] * len(item[1]) * len(item[2]) for item in items]))
            insts = list(chain(*[[itm] * len(item[2]) for item in items for itm in item[1]]))
            tails = list(chain(*[item[2] * len(item[1]) for item in items]))

            prompt_list = [[
                {"role": "user", "content": prompt_template.format('coffee',
                                                                   'If PersonX drinks coffee, as a result, PersonX wants to get refreshed.')},
                {"role": "assistant",
                 "content": "(1) stimulant, (2) energy booster, (3) revitalizing drink, (4) stimulant drink, (5) caffeinated drink, (6) caffeinated beverage, (7) wake-up elixir"},
                {"role": "user", "content": prompt_template.format('PersonX eats less',
                                                                   'If PersonX eats less, as a result, PersonX wants to exercise.')},
                {"role": "assistant",
                 "content": "(1) exercise aid, (2) reduced food intake, (3) limited eating, (4) restricted food consumption"},
                {"role": "user", "content": prompt_template.format('the doctor',
                                                                   'If PersonX takes PersonY to the doctor, as a result, PersonX wants to make sure they are ok.')},
                {"role": "assistant",
                 "content": "(1) expert, (2) professional, (3) medical professional, (4) healthcare provider, (5) healthcare expert, (6) medical practitioner, (7) healthcare specialist, (8) specialist"},
                {"role": "user", "content": prompt_template.format(ins, get_assertion(event, r, tail))},
            ] \
                for ins, event, tail in zip(insts, heads, tails)]
            if len(prompt_list) == 0:
                pred = []
                break

            pred = dispatch_openai_requests(
                messages_list=prompt_list,
                max_tokens=256,
            )

            time.sleep(timeout_succeed)
            break
        except Exception as e:
            print(e)
            time.sleep(timeout_failure)

    temp = '\n'.join([x['choices'][0]['message']['content'] for x in pred])

    output = open(out_filename, 'a+')
    output.write(f'{i}/{total}\n{temp}\n')
    output.close()
    predictions.extend([x['choices'][0]['message']['content'] for x in pred])
