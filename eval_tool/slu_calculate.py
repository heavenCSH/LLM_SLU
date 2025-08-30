import glob
import os
import json
import pandas as pd
import re
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

def load_json(path):
    all_data = []
    with open(path, 'r') as file:
        for line in file:
            data = json.loads(line)
            all_data.append(data)
    return all_data

def save_json(all_data, path):
    folder_path = os.path.dirname(path)
    os.makedirs(folder_path, exist_ok = True)
    with open(path, 'w', encoding = 'utf-8') as file:
        for data in all_data:
            json.dump(data, file, ensure_ascii = False)
            file.write('\n')

def calculate_list_dimension(lst):
    if isinstance(lst, list):
        if not lst:  # 空列表的情况，深度为1
            return 1
        # 遍历每个子元素，递归计算维度
        sub_dimensions = [calculate_list_dimension(item) for item in lst if not isinstance(item, str)]
        return 1 + (max(sub_dimensions) if sub_dimensions else 0)
    return 0  # 非列表元素（包括字符串）

def calculate(eval_model_path, dataset):
    eval_model = eval_model_path.split('/')[-1]
    current_path = "/mnt/nvme0n1/hwb/LLM_SLU/eval_output/{}/{}".format(dataset, eval_model)
    result_pattern = os.path.join(current_path, '*.json')
    result_path_list = glob.glob(result_pattern)
    # /mnt/nvme0n1/hwb/LLM_SLU/eval_output/dataset/model/
    result = []
    for file in result_path_list:
        checkpoint_name = file.split('/')[-1].split('.')[0]
        all_data = load_json(file)
        intent_pred, intent_res = [], []
        slot_pred, slot_res = [], []
        for data in all_data:
            if 'error' in data.keys():
                data['Overall'] = False
                data['Slots'] = ['O'] * len(data['SF'])
                slot_pred.append(['O'] * len(data['SF']))
                slot_res.append(data['SF'])
                continue
            ID = data['ID']
            SF = data['SF']
            Utterance = data['Utterance']
            Intents = data['Intents']
            Entities = data['Entities']
            Slots = ['O'] * len(SF)
            Words = Utterance.split()
            if isinstance(Entities, dict):
                for entity, span_list in Entities.items():
                    if calculate_list_dimension(span_list) == 1: # 防止输出1维列表
                        span_list = [span_list]
                    if isinstance(span_list, list): # 防止span_list不为列表
                        for span in span_list:
                            if not isinstance(span, list) or len(span) == 0:
                                continue
                            starts = [i for i, x in enumerate(Words) if x == span[0]]
                            if len(starts) == 0:
                                continue
                            for start in starts:
                                # start = Words.index(span[0]) if span[0] in Words else -1
                                # if start == -1:
                                #     break
                                Slots[start] = 'B-' + entity
                                for j in range(start + 1, min(start + len(span), len(Slots))):
                                    Slots[j] = 'I-' + entity
            data['Overall'] = True if set(ID) == set(Intents) and SF == Slots else False
            data['Slots'] = Slots
            intent_pred.append(Intents)
            intent_res.append(ID)
            slot_pred.append(Slots)
            slot_res.append(SF)
        save_json(all_data, file)
        acc = sum([1 for i, j in zip(intent_pred, intent_res) if set(i) == set(j)]) / len(all_data)
        f1 = f1_score(slot_res, slot_pred)
        precision = precision_score(slot_res, slot_pred)
        recall = recall_score(slot_res, slot_pred)
        overall = sum([1 for data in all_data if data['Overall'] == True]) / len(all_data)
        result.append([checkpoint_name, acc, f1, overall, precision, recall])
    return result

def fun(checkpoint):
    name, shot_num, _ = checkpoint[0].split('_')
    step = int(name.split('-')[1])
    return (int(shot_num), step)

def save_csv(result, eval_model_path, dataset):
    eval_model = eval_model_path.split('/')[-1]
    save_path = f"/mnt/nvme0n1/hwb/LLM_SLU/eval_output/{dataset}/{eval_model}"
    result = sorted(result, key = fun)
    print(save_path)
    df = pd.DataFrame(result, columns = ["model", "Acc", "F1_score", "Overall", "SF_Precision", "SF_Recall"])
    df.to_csv(save_path + "/result.csv", index = False)

    markdown_data = df.to_markdown(index = False)
    with open(save_path + "/result.md", 'w', encoding = 'utf-8') as file:
        file.write(markdown_data)

if __name__ == '__main__':
    # 评估
    path_list = [
        ["/mnt/nvme0n1/hwb/lora/atis1_2_2shot_Qwen2.5_7B_Instruct", "MixATIS_clean"],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_2shot_Qwen2.5_7B_Instruct", "MixSNIPS_clean"],
        ["/mnt/nvme0n1/hwb/lora/atis1_2_4shot_Qwen2.5_7B_Instruct", "MixATIS_clean"],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_4shot_Qwen2.5_7B_Instruct", "MixSNIPS_clean"],
        ["/mnt/nvme0n1/hwb/lora/atis1_2_6shot_Qwen2.5_7B_Instruct", "MixATIS_clean"],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_6shot_Qwen2.5_7B_Instruct", "MixSNIPS_clean"],
        ["/mnt/nvme0n1/hwb/lora/atis1_2_8shot_Qwen2.5_7B_Instruct", "MixATIS_clean"],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_8shot_Qwen2.5_7B_Instruct", "MixSNIPS_clean"],
        ["/mnt/nvme0n1/hwb/lora/atis1_2_10shot_Qwen2.5_7B_Instruct", "MixATIS_clean"],
        ["/mnt/nvme0n1/hwb/lora/snips1_2_10shot_Qwen2.5_7B_Instruct", "MixSNIPS_clean"]
    ]
    for ele in path_list:
        eval_model_path = ele[0]
        dataset = ele[1]
        result = calculate(eval_model_path, dataset)

        # 保存结果
        save_csv(result, eval_model_path, dataset)