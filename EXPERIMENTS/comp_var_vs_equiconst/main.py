import os
import json
import argparse

import torch
from tqdm import tqdm
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# root directories for the two versions of SEED-Bench dataset
variable_cc3m_dir = "/home/projects/bagon/dannyh/data/seedbench_filt/Variable_10d"
uniform_cc3m_dir = "/home/projects/bagon/dannyh/data/seedbench_filt/Constant_10d"
dimension10_dir = "/home/projects/bagon/shared/20bn-something-something-v2/videos"
# Update these paths as necessary
# dimension11_dir = "/YOUR_PATH_TO/EPIC-KITCHENS/3h91syskeag572hl6tvuovwv4d/videos/test"
# dimension12_dir = "/YOUR_PATH_TO/BreakfastII_15fps_qvga_sync"

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

def visualize_and_save(qa_item, var_img_path, uni_img_path, var_pred, uni_pred, output_dir):
    question = qa_item['question']
    choices = [qa_item['choice_a'], qa_item['choice_b'], qa_item['choice_c'], qa_item['choice_d']]
    gt = qa_item['answer']
    
    var_img = mpimg.imread(var_img_path)
    uni_img = mpimg.imread(uni_img_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(question, fontsize=16)
    axes[0].imshow(var_img)
    axes[0].set_title(f"Variable Res\nPrediction: {var_pred}\nGT: {gt}")
    axes[0].axis('off')
    axes[1].imshow(uni_img)
    axes[1].set_title(f"Uniform Res\nPrediction: {uni_pred}\nGT: {gt}")
    axes[1].axis('off')

    output_path = os.path.join(output_dir, f"{qa_item['question_id']}.png")
    plt.savefig(output_path)
    plt.close(fig)

def run_inference(model, qa_anno, output_dir, variable_cc3m_dir, uniform_cc3m_dir):
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
                var_data_path = os.path.join(variable_cc3m_dir, qa_item['data_id'])
                uni_data_path = os.path.join(uniform_cc3m_dir, qa_item['data_id'])
                try:
                    var_data_path = glob.glob(var_data_path[:-4]+'*')[0]
                    uni_data_path = glob.glob(uni_data_path[:-4]+'*')[0]
                except:
                    print('missing file: ' + var_data_path + ' or ' + uni_data_path)
                    continue
            elif qa_item['data_type'] == 'video':
                if qa_item['question_type_id'] == 10:
                    var_data_path = os.path.join(dimension10_dir, qa_item['data_id'])
                    uni_data_path = var_data_path
                elif qa_item['question_type_id'] == 11:
                    var_data_path = os.path.join(dimension11_dir, qa_item['data_id'])
                    uni_data_path = var_data_path
                    data_info['segment'] = qa_item['segment']
                elif qa_item['question_type_id'] == 12:
                    var_data_path = os.path.join(dimension12_dir, qa_item['data_id'])
                    uni_data_path = var_data_path
                    data_info['segment'] = qa_item['segment']
                else:
                    raise ValueError("The question type id is not valid.")
            else:
                raise ValueError("The data type is not valid.")
            data_info['var_data_path'] = var_data_path
            data_info['uni_data_path'] = uni_data_path

            # Losses: loss values of 4 choices, torch tensor, shape=[4]
            with torch.no_grad():
                var_losses = model({**data_info, 'data_path': var_data_path})
                uni_losses = model({**data_info, 'data_path': uni_data_path})
            var_class_ranks = torch.argsort(var_losses, dim=-1).cpu()
            uni_class_ranks = torch.argsort(uni_losses, dim=-1).cpu()
            var_pred_id = ['A', 'B', 'C', 'D'][var_class_ranks[0]]
            uni_pred_id = ['A', 'B', 'C', 'D'][uni_class_ranks[0]]
            gt = qa_item['answer']
            answer_record = {
                'question_id': qa_item['question_id'],
                'var_prediction': var_pred_id,
                'uni_prediction': uni_pred_id,
                'q_type_id': qa_item['question_type_id'],  # added by DH
                'gt': qa_item['answer']  # added by DH
            }
            answer_list.append(answer_record)
            # Output prediction record for each question
            output_f.write(json.dumps(answer_record) + "\n")
            
            # Visualization
            visualize_and_save(qa_item, var_data_path, uni_data_path, var_pred_id, uni_pred_id, output_dir)
            
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
        var_pred, uni_pred, gt, data_type = item['var_prediction'], item['uni_prediction'], item['gt'], item['q_type_id']

        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        if var_pred == gt:
            correct_counts[(data_type, 'var')] = correct_counts.get((data_type, 'var'), 0) + 1
        if uni_pred == gt:
            correct_counts[(data_type, 'uni')] = correct_counts.get((data_type, 'uni'), 0) + 1

    print("Accuracy for each data type:")
    total_count = 0
    total_var_correct = 0
    total_uni_correct = 0
    for data_type in type_counts.keys():
        var_accuracy = correct_counts.get((data_type, 'var'), 0) / type_counts[data_type] * 100
        uni_accuracy = correct_counts.get((data_type, 'uni'), 0) / type_counts[data_type] * 100
        print(f"Data type {data_type}: Variable Res {var_accuracy:.2f}%, Uniform Res {uni_accuracy:.2f}%")

        total_count += type_counts[data_type]
        total_var_correct += correct_counts.get((data_type, 'var'), 0)
        total_uni_correct += correct_counts.get((data_type, 'uni'), 0)

    total_var_accuracy = total_var_correct / total_count * 100
    total_uni_accuracy = total_uni_correct / total_count * 100
    print(f"Total accuracy: Variable Res {total_var_accuracy:.2f}%, Uniform Res {total_uni_accuracy:.2f}%")

    correct_var_wrong_uni_dir = os.path.join(output_dir, 'correct_var_wrong_uni')
    correct_uni_wrong_var_dir = os.path.join(output_dir, 'correct_uni_wrong_var')

    if not os.path.exists(correct_var_wrong_uni_dir):
        os.mkdir(correct_var_wrong_uni_dir)
    if not os.path.exists(correct_uni_wrong_var_dir):
        os.mkdir(correct_uni_wrong_var_dir)

    for item in answer_list:
        var_pred, uni_pred, gt = item['var_prediction'], item['uni_prediction'], item['gt']
        question_id = item['question_id']
        output_path = os.path.join(output_dir, f"{question_id}.png")
        if var_pred == gt and uni_pred != gt:
            os.system(f"cp {output_path} {correct_var_wrong_uni_dir}")
        if uni_pred == gt and var_pred != gt:
            os.system(f"cp {output_path} {correct_uni_wrong_var_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='instruct_blip')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--task', type=str, default='all')
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
    run_inference(model, qa_anno, args.output_dir, variable_cc3m_dir, uniform_cc3m_dir)
