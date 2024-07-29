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

# root directories for the three versions of SEED-Bench dataset
full_cc3m_dir = "/home/projects/bagon/shared/SEED-Bench/SEED-Bench-image"
variable_cc3m_dir = "/home/projects/bagon/dannyh/data/seedbench_filt/Variable_27d"
uniform_cc3m_dir = "/home/projects/bagon/dannyh/data/seedbench_filt/Constant_27d"
dimension10_dir = "/home/projects/bagon/shared/20bn-something-something-v2/videos"
dimension11_dir = "/YOUR_PATH_TO/EPIC-KITCHENS/3h91syskeag572hl6tvuovwv4d/videos/test"
dimension12_dir = "/YOUR_PATH_TO/BreakfastII_15fps_qvga_sync"

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

def visualize_and_save(qa_item, full_img_path, var_img_path, uni_img_path, full_pred, var_pred, uni_pred, output_dir, perm):
    question = qa_item['question']
    choices = [qa_item['choice_a'], qa_item['choice_b'], qa_item['choice_c'], qa_item['choice_d']]
    gt = qa_item['answer']
    
    full_img = mpimg.imread(full_img_path)
    var_img = mpimg.imread(var_img_path)
    uni_img = mpimg.imread(uni_img_path)
    
    full_pred_text = choices[ord(full_pred) - ord('A')]
    var_pred_text = choices[ord(var_pred) - ord('A')]
    uni_pred_text = choices[ord(uni_pred) - ord('A')]
    gt_text = choices[ord(gt) - ord('A')]

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    fig.suptitle(question, fontsize=16, y=0.95)
    axes[0].imshow(full_img)
    axes[0].set_title("Full", fontsize=14)
    axes[0].axis('off')
    axes[0].text(0.5, -0.15, f"Prediction: {full_pred} ({full_pred_text})\nGT: {gt} ({gt_text})", ha='center', va='top', transform=axes[0].transAxes, fontsize=12)
    axes[1].imshow(var_img)
    axes[1].set_title("Variable", fontsize=14)
    axes[1].axis('off')
    axes[1].text(0.5, -0.15, f"Prediction: {var_pred} ({var_pred_text})\nGT: {gt} ({gt_text})", ha='center', va='top', transform=axes[1].transAxes, fontsize=12)
    axes[2].imshow(uni_img)
    axes[2].set_title("Uniform", fontsize=14)
    axes[2].axis('off')
    axes[2].text(0.5, -0.15, f"Prediction: {uni_pred} ({uni_pred_text})\nGT: {gt} ({gt_text})", ha='center', va='top', transform=axes[2].transAxes, fontsize=12)

    image_id = qa_item['data_id']
    question_id = qa_item['question_id']
    output_path = os.path.join(output_dir, perm, f"{image_id}_{question_id}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def run_inference(model, qa_anno, output_dir, full_cc3m_dir, variable_cc3m_dir, uniform_cc3m_dir):
    total_qa_num = len(qa_anno)
    answer_list = []

    permutations = [
        ('full_right_var_right_uni_right', lambda f, v, u, g: f == g and v == g and u == g),
        ('full_wrong_var_right_uni_right', lambda f, v, u, g: f != g and v == g and u == g),
        ('full_right_var_wrong_uni_right', lambda f, v, u, g: f == g and v != g and u == g),
        ('full_right_var_right_uni_wrong', lambda f, v, u, g: f == g and v == g and u != g),
        ('full_wrong_var_wrong_uni_right', lambda f, v, u, g: f != g and v != g and u == g),
        ('full_wrong_var_right_uni_wrong', lambda f, v, u, g: f != g and v == g and u != g),
        ('full_right_var_wrong_uni_wrong', lambda f, v, u, g: f == g and v != g and u != g),
        ('full_wrong_var_wrong_uni_wrong', lambda f, v, u, g: f != g and v != g and u != g),
    ]

    for perm in permutations:
        perm_dir = os.path.join(output_dir, perm[0])
        if not os.path.exists(perm_dir):
            os.makedirs(perm_dir)

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
                full_data_path = os.path.join(full_cc3m_dir, qa_item['data_id'])
                var_data_path = os.path.join(variable_cc3m_dir, qa_item['data_id'])
                uni_data_path = os.path.join(uniform_cc3m_dir, qa_item['data_id'])
                try:
                    full_data_path = glob.glob(full_data_path[:-4]+'*')[0]
                    var_data_path = glob.glob(var_data_path[:-4]+'*')[0]
                    uni_data_path = glob.glob(uni_data_path[:-4]+'*')[0]
                except:
                    print('missing file: ' + full_data_path + ' or ' + var_data_path + ' or ' + uni_data_path)
                    continue
            elif qa_item['data_type'] == 'video':
                if qa_item['question_type_id'] == 10:
                    full_data_path = os.path.join(dimension10_dir, qa_item['data_id'])
                    var_data_path = full_data_path
                    uni_data_path = full_data_path
                elif qa_item['question_type_id'] == 11:
                    full_data_path = os.path.join(dimension11_dir, qa_item['data_id'])
                    var_data_path = full_data_path
                    uni_data_path = full_data_path
                    data_info['segment'] = qa_item['segment']
                elif qa_item['question_type_id'] == 12:
                    full_data_path = os.path.join(dimension12_dir, qa_item['data_id'])
                    var_data_path = full_data_path
                    uni_data_path = full_data_path
                    data_info['segment'] = qa_item['segment']
                else:
                    raise ValueError("The question type id is not valid.")
            else:
                raise ValueError("The data type is not valid.")
            data_info['full_data_path'] = full_data_path
            data_info['var_data_path'] = var_data_path
            data_info['uni_data_path'] = uni_data_path

            # Losses: loss values of 4 choices, torch tensor, shape=[4]
            with torch.no_grad():
                full_losses = model({**data_info, 'data_path': full_data_path})
                var_losses = model({**data_info, 'data_path': var_data_path})
                uni_losses = model({**data_info, 'data_path': uni_data_path})
            full_class_ranks = torch.argsort(full_losses, dim=-1).cpu()
            var_class_ranks = torch.argsort(var_losses, dim=-1).cpu()
            uni_class_ranks = torch.argsort(uni_losses, dim=-1).cpu()
            full_pred_id = ['A', 'B', 'C', 'D'][full_class_ranks[0]]
            var_pred_id = ['A', 'B', 'C', 'D'][var_class_ranks[0]]
            uni_pred_id = ['A', 'B', 'C', 'D'][uni_class_ranks[0]]
            gt = qa_item['answer']
            answer_record = {
                'question_id': qa_item['question_id'],
                'full_prediction': full_pred_id,
                'var_prediction': var_pred_id,
                'uni_prediction': uni_pred_id,
                'q_type_id': qa_item['question_type_id'],
                'gt': qa_item['answer']
            }
            answer_list.append(answer_record)
            # Output prediction record for each question
            output_f.write(json.dumps(answer_record) + "\n")
            
            # Determine permutation folder and save visualization
            for perm in permutations:
                if perm[1](full_pred_id, var_pred_id, uni_pred_id, gt):
                    visualize_and_save(qa_item, full_data_path, var_data_path, uni_data_path, full_pred_id, var_pred_id, uni_pred_id, output_dir, perm[0])
            
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
        full_pred, var_pred, uni_pred, gt, data_type = item['full_prediction'], item['var_prediction'], item['uni_prediction'], item['gt'], item['q_type_id']

        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        if full_pred == gt:
            correct_counts[(data_type, 'full')] = correct_counts.get((data_type, 'full'), 0) + 1
        if var_pred == gt:
            correct_counts[(data_type, 'var')] = correct_counts.get((data_type, 'var'), 0) + 1
        if uni_pred == gt:
            correct_counts[(data_type, 'uni')] = correct_counts.get((data_type, 'uni'), 0) + 1

    print("Accuracy for each data type:")
    total_count = 0
    total_full_correct = 0
    total_var_correct = 0
    total_uni_correct = 0
    for data_type in type_counts.keys():
        full_accuracy = correct_counts.get((data_type, 'full'), 0) / type_counts[data_type] * 100
        var_accuracy = correct_counts.get((data_type, 'var'), 0) / type_counts[data_type] * 100
        uni_accuracy = correct_counts.get((data_type, 'uni'), 0) / type_counts[data_type] * 100
        print(f"Data type {data_type}: Full Res {full_accuracy:.2f}%, Variable Res {var_accuracy:.2f}%, Uniform Res {uni_accuracy:.2f}%")

        total_count += type_counts[data_type]
        total_full_correct += correct_counts.get((data_type, 'full'), 0)
        total_var_correct += correct_counts.get((data_type, 'var'), 0)
        total_uni_correct += correct_counts.get((data_type, 'uni'), 0)

    total_full_accuracy = total_full_correct / total_count * 100
    total_var_accuracy = total_var_correct / total_count * 100
    total_uni_accuracy = total_uni_correct / total_count * 100
    print(f"Total accuracy: Full Res {total_full_accuracy:.2f}%, Variable Res {total_var_accuracy:.2f}%, Uniform Res {total_uni_accuracy:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='blip2')
    parser.add_argument('--anno_path', type=str, default='/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Programming/BLIP2/EXPERIMENTS/comp_var_vs_equiconst/SEED-Bench.json')
    parser.add_argument('--output_dir', type=str, default='/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Programming/BLIP2/EXPERIMENTS/comp_var_vs_equiconst/visualizations_27d')
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--gpu', type=int, default=1, help='GPU device ID')
    args = parser.parse_args()
    
    qa_anno = json.load(open(args.anno_path, 'rb'))
    if 'questions' in qa_anno.keys():
        qa_anno = qa_anno['questions']
    qa_anno = filter_questions(qa_anno, args.task)

    print('Output dir: '+args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print(f'evaluating.. {args.model} on GPU {args.gpu}')
    # The interface for testing MLLMs
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model).to(device)
    run_inference(model, qa_anno, args.output_dir, full_cc3m_dir, variable_cc3m_dir, uniform_cc3m_dir)
