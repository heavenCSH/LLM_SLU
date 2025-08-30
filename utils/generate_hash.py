import hashlib, json

TRAIN_DATASET_INFO_PATH = "/mnt/nvme0n1/hwb/LLM_SLU/LLaMAFactory/data/dataset_info.json"

def generate_hash_for_dataset(save_path: str, version: str):
    with open(save_path, 'rb') as datafile:
        binary_data = datafile.read()
    sha1 = hashlib.sha1(binary_data).hexdigest()
    with open(TRAIN_DATASET_INFO_PATH, 'r') as f:
        dataset_info = json.load(f)
    dataset_info[version] = {
        "file_name": save_path,
        "file_sha1": sha1,
        "columns": {
            "system": "system",
            "prompt": "prompt",
            "response": "response",
            "history": "history"
        }
    }
    with open(TRAIN_DATASET_INFO_PATH, 'w') as f:
        json.dump(dataset_info, f, ensure_ascii = False, indent = 4)

def main():
    shot_num = [2, 4, 6, 8, 10]
    for shot in shot_num:
        generate_hash_for_dataset('/mnt/nvme0n1/hwb/LLM_SLU/data/LM1_1FewShotMixATIS/fewshot-{}/{}-shot-data.json'.format(shot, shot), "atis1_1_{}_shot".format(shot))
    for shot in shot_num:
        generate_hash_for_dataset('/mnt/nvme0n1/hwb/LLM_SLU/data/LM1_1FewShotMixSNIPS/fewshot-{}/{}-shot-data.json'.format(shot, shot), "snips1_1_{}_shot".format(shot))

if __name__ == "__main__":
    main()