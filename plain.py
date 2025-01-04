import argparse
import os
import json
import re
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList
import sys
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image
import re

sys.path.append("/home/projects/MiniGPT4")

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import MyChat, StoppingCriteriaSub

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from dataclasses import dataclass
# from metrics import test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="vicuna_7b", type=str, help='model_name')
    parser.add_argument('--method', default='regenerate', type=str)
    parser.add_argument('--rewrite', action='store_true', default=False)
    parser.add_argument('--device', default=0, type=int, help='the device gpu or cpu')
    parser.add_argument('--seed', default=13, type=int, help='seed')
    parser.add_argument('--task', default='mme_color', type=str)
    parser.add_argument('--temperature', default=1, type=float)

    return parser.parse_args()

model_name_to_config = {
    "vicuna_13b": "/home/projects/MiniGPT4/eval_configs/minigpt4_vicuna13b_eval.yaml",
    "vicuna_7b": "/home/projects/MiniGPT4/eval_configs/minigpt4_vicuna7b_eval.yaml",
    "llama2_7b": "/home/projects/MiniGPT4/eval_configs/minigpt4_llama2_eval.yaml"
}

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def parse_response(response):
    response = response.strip()
    response = response.strip("#")

    return response

def load_data(args):
    datas = []
    print(args.task)
    print(args.method)
    if "regenerate" not in args.method:
        if args.task == "qa90":
            data_dir = "/data/datas/datas/coco_val2014/"
            test_file = data_dir + "qa90.jsonl"

            with open(test_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    datas.append(json.loads(line))
        elif "mme" in args.task:
            sub_task = args.task[4:]
            test_file = f"/data/datas/datas/MME/eval_tool/Your_Results/{sub_task}.txt"
            image_dir = f"/data/datas/datas/MME/{sub_task}/"
            idx = 0
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    image, text, answer = line.strip().split('\t')
                    data = {"id": idx, "image": image_dir + image, "text": text, "answer": answer}
                    idx += 1
                    datas.append(data)

        elif args.task == "amber" :
            test_file = "/data/datas/datas/AMBER/query_generative_random_200.jsonl"
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    data["text"] = data["query"]
                    datas.append(data)
        elif "pope" in args.task:
            test_file = f"/data/datas/datas/pope/coco_{args.task}.json"
            image_dir = "/data/datas/datas/pope/pope_image/"
            idx = 0
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    data["id"] = idx
                    idx += 1
                    data["image"] = image_dir + data["image"]
                    datas.append(data)
        
    else:
        # test_file = "/home/projects/multimodal_dialog_generation/add_information/result/" + args.model_name + "_" + args.method + "/13/" + args.task + "_answer.json"
        test_file = "/home/projects/multimodal_dialog_generation/add_information/result/" + args.model_name + "_" + args.method + "/13/" + args.task + "_answer.json"

        if "mme" in args.task:
            sub_task = args.task[4:]
            image_dir = f"/data/datas/datas/MME/{sub_task}/"
            idx = 0
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    idx += 1
                    datas.append(data)
        elif args.task == "amber" :
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    datas.append(data)
        elif "pope" in args.task:
            image_dir = "/data/datas/datas/pope/pope_image/"
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    datas.append(data)

    return datas

def process_result(args):
    output_dir = "/home/projects/multimodal_dialog_generation/add_information/result/" + args.model_name + "_" + args.method + "/" + str(args.seed) + "/"
    output_file = output_dir + args.task + ".json"
    

    ids = set()
    res = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if not args.rewrite:
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding="utf-8") as f:
                    for line in f.readlines():
                        if line:
                            data = json.loads(line)
                            ids.add(data["id"])
                            res.append(data)

    with open(output_file, 'w', encoding="utf-8") as f:
        for data in res:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    with open(output_dir + args.task + "_args.json", "w", encoding="utf-8") as f:
        f.write(str(args))

    return ids, output_file

def load_model(model_args):
    print('Initializing Chat')
    args = model_args
    cfg = Config(model_args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = MyChat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    return chat

def process_information(single_claim, multiple_claim, args):
    single_claims = single_claim.split("Single Object Level:")
    if len(single_claim) == 2:
        count_claim = single_claims[0]
        single_claim = "Single Object Level:" + single_claims[1]
    else:
        count_claim = single_claims[0]
        single_claim = ""
    
    count_claim = count_claim.strip()
    single_claim = single_claim.strip()
    multiple_claim = multiple_claim.strip()
    
    if "no_multiple" in args.method:
        multiple_claim = ""
    elif "no_count" in args.method:
        count_claim = ""
    elif "no_single_attribute" in args.method:
        single_claim = ""

    
    sup_info = ""
    if count_claim != "":
        sup_info += count_claim + "\n"
    if single_claim != "":
        sup_info += single_claim + "\n"
    if multiple_claim != "":
        sup_info += multiple_claim + "\n"
    
    return sup_info



def get_claim(sample, args):

    single_claim_string = sample['single_claim_string']
    multiple_claim_string = sample['multiple_claim_string']

    single_claim_string_list = single_claim_string.split("\n")
    multiple_claim_string_list = multiple_claim_string.split("\n")
    if len(single_claim_string_list) > 50:
        return "" 
    else:
        single_claim_string = "\n".join(single_claim_string_list) + "\n"
    if len(multiple_claim_string_list) > 50:
        return "" 
    else:
        multiple_claim_string = "\n".join(multiple_claim_string_list) + "\n"


    
    sup_info = ""
    if single_claim_string != "":
        sup_info += single_claim_string
    if multiple_claim_string != "":
        sup_info += multiple_claim_string
    
    return sup_info


def get_base_response(chat, question, image, args):
    base_prompt = '''Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions. ###Human: <Img><ImageHere></Img> {question}###Assistant:'''

    inputs_json = {"question": question}
    inputs = base_prompt.format(**inputs_json)
    img_list = [image]

    predict = chat.answer(inputs, img_list, temperature=args.temperature)
    parse_response = predict.strip().strip("#").strip()
    parse_response = parse_response.replace('\\n', '')

    output_dict = {}
    output_dict["inputs"] = inputs
    output_dict["predict"] = predict
    output_dict["parse_response"] = parse_response

    return output_dict


def get_response_from_information(chat, information, question, image, args):
    if information != "":
        base_prompt = '''Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions. ###Human: <Img><ImageHere></Img> ###Human: I am a visual expert and extract some information from the image, you are required to answer my questions based on the information, following these rules:
1. The supplementary information may contain some of the following parts:
    Counting: the number of occurrences for specific types of entities;
    Single Object Level: describes the attributes of single entity instance, such as bounding boxes, colors, etc. The attributes is formatted as "entity 1: [bbox] "attributes of this entity".
    Multiple Objects Level: relationship between multiple entity objects, such as positional relationship. 
2. Your response must correspond with the supplementary information provided.
3. When considering about the exists of objects or the number of objects , it is necessary to maintain consistency with the "Counting" information.
4. When considering about the attribute of objects such as color, it is necessary to maintain consistency with the "Single Object Level" information.
5. When considering about the positional relationship between objects, it is necessary to maintain consistency with the "Multiple Objects Level" information.
6. Ensure your answers are relevant to the question.

Supplementary information:
{information} Please answer my questions based on the supplementary above. ###Human: {question} ###Assistant:'''

        inputs_json = {"question": question, "information": information}
        inputs = base_prompt.format(**inputs_json)
    else:
        base_prompt = '''Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions. ###Human: <Img><ImageHere></Img> {question}###Assistant:'''
        inputs_json = {"question": question}
        inputs = base_prompt.format(**inputs_json)

    img_list = [image]

    predict = chat.answer(inputs, img_list, temperature=args.temperature)
    parse_response = predict.strip().strip("#").strip()
    parse_response = parse_response.replace('\\n', '')

    output_dict = {}
    output_dict["inputs"] = inputs
    output_dict["predict"] = predict
    output_dict["parse_response"] = parse_response

    return output_dict


def get_response(datas, ids, output_file, args):
    for i in tqdm(range(len(datas))):
    # for i in tqdm(range(10)):
        print("idx:", i)
        example = datas[i]
        if example["id"] in ids:
            continue
        question = example["text"]
        image = example["image"]
        print("get base response")
        base_response_dict = get_base_response(chat, question, image, args)
        base_response = base_response_dict["parse_response"]

        output_dict = example
        output_dict["base_response_dict"] = base_response_dict
        output_dict["final_response"] = base_response

        with open(output_file, "a+", encoding="utf-8") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False)+'\n')


def get_response_with_information(datas, ids, output_file, args):
    for i in tqdm(range(len(datas))):
    # for i in tqdm(range(10)):
        print("idx:", i)
        example = datas[i]
        if example["id"] in ids:
            continue
        question = example["query"]
        image = example["img_path"]
        if example["named_entity"] and example["named_entity"][0] == "None":
            information = ""
        else:
            information = get_claim(example, args)
        print("get response with information")
        if information == "":
            base_response = example["input_desc"]
        else:
            base_response_dict = get_response_from_information(chat, information, question, image, args)
            base_response = base_response_dict["parse_response"]

        output_dict = example
        # output_dict["base_response_dict"] = base_response_dict
        output_dict["final_response"] = base_response

        with open(output_file, "a+", encoding="utf-8") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False)+'\n')

if __name__ == "__main__":
    args = get_args()
    # args.device = "cuda:"+str(args.device)
    model_name = args.model_name
    
    datas = load_data(args)
    ids, output_file = process_result(args)

    @dataclass
    class Args:
        cfg_path: str = model_name_to_config[model_name]
        gpu_id: int = args.device
        options = None

    model_args = Args()
    chat = load_model(model_args)
    setup_seeds(args.seed)
    if args.method == "plain":
        get_response(datas, ids, output_file, args)
    elif "regenerate" in args.method:
        get_response_with_information(datas, ids, output_file, args)

