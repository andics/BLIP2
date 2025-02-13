import os
import json
import argparse

import torch
from tqdm import tqdm
import numpy as np
import random
import glob # added by DH

# root directory of evaluation dimension 1-9
cc3m_dir = "/home/projects/bagon/dannyh/data/seedbench_filt/Variable_10d"
# root directory of evaluation dimension 10
dimension10_dir = "/home/projects/bagon/shared/20bn-something-something-v2/videos"
# root directory of evaluation dimension 11
#dimension11_dir = "/YOUR_PATH_TO/EPIC-KITCHENS/3h91syskeag572hl6tvuovwv4d/videos/test"
# root directory of evaluation dimension 12
#dimension12_dir = "/YOUR_PATH_TO/BreakfastII_15fps_qvga_sync"

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def filter_questions(data, task='all'):
    if task == "image":
        return [q for q in data if 1 <= q["question_type_id"] <= 9]
    elif task == "video":
        return [q for q in data if 10 <= q["question_type_id"] <= 12]
    elif task == "all":
        return data
    elif is_integer_string(task):
        return [q for q in data if q["question_type_id"] == int(task)]
    else:
        raise ValueError(f"Invalid task: {task}")


def build_model(model_name):
    if model_name == 'instruct_blip':
        from model.instruct_blip_interface import build
    if model_name == 'blip2':
        from model.blip2_interface import build
    if model_name == 'huggingface_instruct_blip':
        from model.huggingface_instructblip_interface import build

    model = build()

    return model

def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def filter_questions(data, task='all'):
    if task == "image":
        return [q for q in data if 1 <= q["question_type_id"] <= 9]
    elif task == "video":
        return [q for q in data if 10 <= q["question_type_id"] <= 12]
    elif task == "all":
        return data
    elif is_integer_string(task):
        return [q for q in data if q["question_type_id"] == int(task)]
    else:
        raise ValueError(f"Invalid task: {task}")


def run_inference(model, qa_anno, output_dir):
    total_qa_num = len(qa_anno)
    answer_list = []
    if not os.path.exists(os.path.join(output_dir, "results.json")):
        output_f = open(os.path.join(output_dir, "results.json"), "w")
        step = 0
        for qa_item in tqdm(qa_anno):

            data_info = {
                'question': qa_item['question'],
                'choices': [qa_item['choice_a'], qa_item['choice_b'], qa_item['choice_c'], qa_item['choice_d']],
                'data_type': qa_item['data_type'],
            }

            if qa_item['data_type'] == 'image':
                data_path = os.path.join(cc3m_dir, qa_item['data_id'])
                # added by DH
                try:
                    data_path = glob.glob(data_path[:-4]+'*')[0]
                except:
                    print('missing file: ' + data_path)
                    continue
            elif qa_item['data_type'] == 'video':
                if qa_item['question_type_id'] == 10:
                    data_path = os.path.join(dimension10_dir, qa_item['data_id'])
                elif qa_item['question_type_id'] == 11:
                    data_path = os.path.join(dimension11_dir, qa_item['data_id'])
                    data_info['segment'] = qa_item['segment']
                elif qa_item['question_type_id'] == 12:
                    data_path = os.path.join(dimension12_dir, qa_item['data_id'])
                    data_info['segment'] = qa_item['segment']
                else:
                    raise ValueError("The question type id is not valid.")
            else:
                raise ValueError("The data type is not valid.")
            data_info['data_path'] = data_path

            # losses: loss values of 4 choices, torch tensor, shape=[4]
            with torch.no_grad():
                losses = model(data_info)
            class_ranks = torch.argsort(losses, dim=-1).cpu()
            pred_id = ['A', 'B', 'C', 'D'][class_ranks[0]]
            gt = qa_item['answer']
            answer_record = {
                'question_id': qa_item['question_id'],
                'prediction': pred_id,
                'q_type_id': qa_item['question_type_id'],  # added by DH
                'gt': qa_item['answer']  # added by DH
            }
            answer_list.append(answer_record)
            # output prediction record for each question
            output_f.write(json.dumps(answer_record) + "\n")
            step += 1

        print("evaluation finished! Calculating accuracy...")
    else:
        print("Loading results from JSON file. Calculating accuracy...")
        with open(os.path.join(output_dir, "results.json"),  mode="r", encoding="utf-8") as f:
            while True:
                ans = f.readline()
                if ans == '':
                    break
                answer_list.append(json.loads(ans))

    type_counts = {}
    correct_counts = {}

    for item in answer_list:
        pred, gt, data_type = item['prediction'], item['gt'], item['q_type_id']

        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        if pred == gt:
            correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

    print("Accuracy for each data type:")
    total_count = 0
    total_correct = 0
    for data_type in type_counts.keys():
        accuracy = correct_counts[data_type] / type_counts[data_type] * 100
        print(f"Data type {data_type}: {accuracy:.2f}%")

        total_count += type_counts[data_type]
        total_correct += correct_counts[data_type]

    total_accuracy = total_correct / total_count * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='instruct_blip')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--task', type=str, default='all')
    args = parser.parse_args()
    
    args = parser.parse_args()

    qa_anno = json.load(open(args.anno_path, 'rb'))
    if 'questions' in qa_anno.keys():
        qa_anno = qa_anno['questions']
    qa_anno = filter_questions(qa_anno, args.task)

    print('Output dir: '+args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print(f'evaluating.. {args.model}')
    # The interface for testing MLLMs
    model = build_model(args.model).cuda()
    run_inference(model, qa_anno, args.output_dir)
