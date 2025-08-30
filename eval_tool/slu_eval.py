import json, json5
import glob
from tqdm import tqdm
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
from LLaMAFactory.src.llamafactory.data import get_template_and_fix_tokenizer
from LLaMAFactory.src.llamafactory.hparams.data_args import DataArguments

atis_intents = []
with open('/mnt/nvme0n1/hwb/LLM_SLU/data/MixATIS_clean/vocab/intent_vocab', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        atis_intents.append(line.strip().split('=-=')[0])

atis_entities = set()
atis_entities.add('O')
with open('/mnt/nvme0n1/hwb/LLM_SLU/data/MixATIS_clean/vocab/slot_vocab', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        if len(line.strip().split('-')) > 1:
            atis_entities.add(line.strip().split('-')[1])

output_format = {
    "Intents": ["intent1", "intent2"],
    "Entities": {
        "entity1": [["word1", "word2"], ["word3", "word4"]],
        "entity2": [["word5"]]
    }
}
example = {
    "Utterance": "what city is mco and what's restriction ap68"
}
example_output = {
    "Intents": ["atis_city", "atis_restriction"],
    "Entities": {
        "airport_code": [["mco"]],
        "restriction_code": [["ap68"]]
    }
}
atis_system_prompt = {
    "Persona": "You are an expert in multi-intent spoken language understanding. Your task is to extract all possible intents and named entities from user utterances while strictly following guidelines for quality and formatting.",
    "Instructions": [
        "You will be given a user utterance.",
        "Let's think step by step. First, identify the intents in the utterance. The intent options are: {}.".format(atis_intents),
        "Next, identify the named entities in the utterance. The named entity options are: {}.".format(list(sorted(atis_entities))),
        "If an entity appears multiple times in the utterance, list all the words that belong to the entity.",
        "Make sure not to output any extra content."
    ],
    "OutputFormat": f"{json.dumps(output_format, ensure_ascii = False, indent = 4)}",
    "Example": f"{json.dumps(example, ensure_ascii = False, indent = 4)}\n{json.dumps(example_output, ensure_ascii = False, indent = 4)}",
}

snips_intents = []
with open('/mnt/nvme0n1/hwb/LLM_SLU/data/MixSNIPS_clean/vocab/intent_vocab', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        snips_intents.append(line.strip().split('=-=')[0])

snips_entities = set()
snips_entities.add('O')
with open('/mnt/nvme0n1/hwb/LLM_SLU/data/MixSNIPS_clean/vocab/slot_vocab', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        if len(line.strip().split('-')) > 1:
            snips_entities.add(line.strip().split('-')[1])

output_format = {
    "Intents": ["intent1", "intent2"],
    "Entities": {
        "entity1": [["word1", "word2"], ["word3", "word4"]],
        "entity2": [["word5"]]
    }
}

example = {
    "Utterance": "mark this album a score of 5 and also what are the movie times"
}
example_output = {
    "Intents": ["RateBook", "SearchScreeningEvent"],
    "Entities": {
        "object_select": [["this"]],
        "object_type": [["album"], ["movie", "times"]],
        "rating_value": [["5"]],
    }
}
snips_system_prompt = {
    "Persona": "You are an expert in multi-intent spoken language understanding. Your task is to extract all possible intents and named entities from user utterances while strictly following guidelines for quality and formatting.",
    "Instructions": [
        "You will be given a user utterance.",
        "Let's think step by step. First, identify the intents in the utterance. The intent options are: {}.".format(snips_intents),
        "Next, identify the named entities in the utterance. The named entity options are: {}.".format(list(sorted(snips_entities))),
        "If an entity appears multiple times in the utterance, list all the words that belong to the entity.",
        "Make sure not to output any extra content."
    ],
    "OutputFormat": f"{json.dumps(output_format, ensure_ascii = False, indent = 4)}",
    "Example": f"{json.dumps(example, ensure_ascii = False, indent = 4)}\n{json.dumps(example_output, ensure_ascii = False, indent = 4)}",
}
system_prompt = [f"{json.dumps(atis_system_prompt, ensure_ascii = False, indent = 4)}", f"{json.dumps(snips_system_prompt, ensure_ascii = False, indent = 4)}"]

device = "auto"
def load_model(model_path):
    is_peft = os.path.exists(os.path.join(
        model_path, "adapter_config.json"))
    if is_peft:
    # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype = torch.float16, device_map = device, trust_remote_code = True)
        model = PeftModel.from_pretrained(
            base_model, model_path, device_map = device, trust_remote_code = True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype = torch.float16, device_map = device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code = True)
    tokenizer.padding_side = "left"  # 修正填充方式
    model = model.merge_and_unload()
    return model, tokenizer

def infer(messages, model, tokenizer, template_name):
    data_args = DataArguments(template = template_name)
    get_template_and_fix_tokenizer(tokenizer, data_args)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )   
    terminators = [tokenizer.eos_token_id]
    model_inputs = tokenizer([text], return_tensors = "pt").to(model.device)
    generated_ids = model.generate(**model_inputs, eos_token_id = terminators, max_new_tokens = 512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)[0]
    return response

def infer_batch(messages_batch, model, tokenizer, template_name):
    data_args = DataArguments(template = template_name)
    get_template_and_fix_tokenizer(tokenizer, data_args)
    texts = [tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    ) for messages in messages_batch]
    
    terminators = [tokenizer.eos_token_id]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(**model_inputs, eos_token_id=terminators, max_new_tokens=512)
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    results = []
    for response in responses:
        pattern = r'\{\s*"Intents":\s*\[.*?\],\s*"Entities":\s*\{.*?\}\s*\}'
        matches = re.findall(pattern, response, re.DOTALL)
        valid_json = None
        for match in matches:
            try:
                candidate = json5.loads(match.strip())
                if "Intents" in candidate and "Entities" in candidate:
                    valid_json = candidate
                    break
            except Exception as e:
                continue
        results.append(valid_json if valid_json else {'error': response})
    return results

def save_json(all_data, path):
    folder_path = os.path.dirname(path)
    os.makedirs(folder_path, exist_ok = True)
    with open(path, 'w', encoding = 'utf-8') as file:
        for data in all_data:
            json.dump(data, file, ensure_ascii = False)
            file.write('\n')

def main(current_path, template_name, dataset, shot, system_prompt):
    current_model = current_path.split('/')[-1]
    checkpoint_pattern = os.path.join(current_path, 'checkpoint*')

    # 加载测试数据
    val_path = f"/mnt/nvme0n1/hwb/LLM_SLU/data/{dataset}/test/seq.in"
    label_path = f"/mnt/nvme0n1/hwb/LLM_SLU/data/{dataset}/test/seq.out"
    with open(val_path, 'r') as f, open(label_path, 'r') as f2:
        utterances = f.read().splitlines()
        labels = f2.read().splitlines()
    
    model_path_list = glob.glob(checkpoint_pattern)
    for model_path in tqdm(model_path_list, desc="Models"):
        checkpoint_name = model_path.split('/')[-1]
        model, tokenizer = load_model(model_path)
        results = []

        batch_size = 8  # 每批次推理的条数
        for i in tqdm(range(0, len(utterances), batch_size), desc="Data", leave=False):
            batch_utterances = utterances[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            messages_batch = [
                [
                    {'content': system_prompt, 'role': 'system'},
                    {'content': f"{json.dumps({'Utterance': utterance}, ensure_ascii=False, indent=4)}", 'role': 'user'}
                ]
                for utterance in batch_utterances
            ]

            raw_responses = infer_batch(messages_batch, model, tokenizer, template_name)
            
            for utterance, label, res in zip(batch_utterances, batch_labels, raw_responses):
                # is_match = re.search(r"```json\n(.*?)\n```", raw_res, re.DOTALL)
                # if is_match:
                #     try:
                #         res = json.loads(is_match.group(1))
                #     except:
                #         res = {'error': raw_res}  # json解析错误
                # else:
                #     res = {'error': raw_res}  # 正则解析错误
                
                if 'error' not in res.keys():
                    res['Intents'] = list(set(res['Intents']))
                res['Utterance'] = utterance.strip()
                res['ID'] = label.strip().split(' ')[0].split('#')  # intent label
                res['SF'] = label.strip().split(' ')[1:]  # slot label
                results.append(res)

        save_json(results, f"/mnt/nvme0n1/hwb/LLM_SLU/eval_output/{dataset}/{current_model}/{checkpoint_name}_{shot}_shot.json")
        print(f"/mnt/nvme0n1/hwb/LLM_SLU/eval_output/{dataset}/{current_model}/{checkpoint_name}_{shot}_shot.json")



if __name__=="__main__":
    path_list = [
        ["/mnt/nvme0n1/hwb/lora/atis1_2_2shot_Qwen2.5_7B_Instruct", "qwen", "MixATIS_clean", 2, system_prompt[0]],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_2shot_Qwen2.5_7B_Instruct", "qwen", "MixSNIPS_clean", 2, system_prompt[1]],
        ["/mnt/nvme0n1/hwb/lora/atis1_2_4shot_Qwen2.5_7B_Instruct", "qwen", "MixATIS_clean", 4, system_prompt[0]],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_4shot_Qwen2.5_7B_Instruct", "qwen", "MixSNIPS_clean", 4, system_prompt[1]],
        ["/mnt/nvme0n1/hwb/lora/atis1_2_6shot_Qwen2.5_7B_Instruct", "qwen", "MixATIS_clean", 6, system_prompt[0]],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_6shot_Qwen2.5_7B_Instruct", "qwen", "MixSNIPS_clean", 6, system_prompt[1]],
        ["/mnt/nvme0n1/hwb/lora/atis1_2_8shot_Qwen2.5_7B_Instruct", "qwen", "MixATIS_clean", 8, system_prompt[0]],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_8shot_Qwen2.5_7B_Instruct", "qwen", "MixSNIPS_clean", 8, system_prompt[1]],
        ["/mnt/nvme0n1/hwb/lora/atis1_2_10shot_Qwen2.5_7B_Instruct", "qwen", "MixATIS_clean", 10, system_prompt[0]],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_10shot_Qwen2.5_7B_Instruct", "qwen", "MixSNIPS_clean", 10, system_prompt[1]]
    ]
    for ele in path_list:
        current_path = ele[0] # lora权重路径
        template_name = ele[1] # 模型chat template名称
        dataset = ele[2] # 数据集
        shot = ele[3] # shot num
        system_prompt = ele[4] # system prompt
        main(current_path, template_name, dataset, shot, system_prompt)