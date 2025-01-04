import sys
import spacy
from typing import List, Dict
import openai
import time
from tqdm import tqdm
from types import SimpleNamespace
sys.path.append("/home/projects/Hallucination/Woodpecker_regenerate_new")
from models.preprocessor import PreProcessor
from models.entity_extractor import EntityExtractor
from models.detector import Detector
from models.questioner import Questioner
from models.answerer import Answerer
from models.claim_generator import ClaimGenerator
from models.refiner import Refiner
from tqdm import tqdm
from typing import List, Dict
import argparse
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList
import sys
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import json
import os

openai.api_base = "https://openkey.cloud/v1" # 
# openai.api_key = "API_KEY"
openai.api_key = "****"

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="vicuna_13b", type=str, help='model_name')
    parser.add_argument('--method', default='woodpecker', type=str)
    parser.add_argument('--rewrite', action='store_true', default=False)
    parser.add_argument('--device', default=0, type=int, help='the device gpu or cpu')
    parser.add_argument('--seed', default=21, type=int, help='seed')
    parser.add_argument('--task', default='mme_color', type=str)
    parser.add_argument('--mode', default='entity_extract', type=str)

    return parser.parse_args()


def load_data(args):
    datas = []
    if args.mode == "entity_extract":
        source_template = "/home/projects/multimodal_dialog_generation/add_information/result/" + args.model_name + "_regenerate/" +  args.task + "{}.json"
    else:
        # source_template = "/home/projects/multimodal_dialog_generation/add_information/result/" + args.model_name + "_" + args.method + "_best/" +  args.task + "{}.json"
        source_template = "/home/projects/multimodal_dialog_generation/add_information/result/" + args.model_name + "_" + args.method + "/" + str(args.seed) + "/" + args.task + "{}.json"

    MODES = {
        "entity_extract": "",
        "detecte": "_entity_extract",
        "question": "_detecte",
        "answer": "_question"
    }
    source_file = source_template.format(MODES[args.mode])
    if args.task == "qa90":
        data_dir = "/data/datas/datas/coco_val2014/"
        test_file = data_dir + "qa90.jsonl"

        with open(test_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                datas.append(json.loads(line))
    elif "mme" in args.task:
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line:
                    data = json.loads(line)
                    datas.append(data)
    # exit()
    elif args.task == "amber" :
        test_file = "/data/datas/datas/AMBER/query_generative_random_200.jsonl"
        origins = []
        idx = 0
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line:
                    origins.append(json.loads(line))

        with open(test_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line)
                origin_json = origins[idx]
                idx += 1
                data["img_path"] = data["image"]
                data["input_desc"] = origin_json["final_response"]
                datas.append(data)
    elif "pope" in args.task:
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line:
                    data = json.loads(line)
                    datas.append(data)

    return datas

def process_result(args):
    output_dir = "/home/projects/multimodal_dialog_generation/add_information/result/" + args.model_name + "_" + args.method + "/" + str(args.seed) + "/"
    output_file = output_dir + args.task + "_" + args.mode + ".json"
    
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

    with open(output_dir + args.task + "_" + args.mode + "_args.json", "w", encoding="utf-8") as f:
        f.write(str(args))

    return ids, output_file


class Corrector:
    def __init__(self, args) -> None:
        # init all the model
        self.args = args
        if args.mode == "entity_extract":
            # self.preprocessor = PreProcessor(args)
            self.entity_extractor = EntityExtractor(args)
        elif args.mode == "detecte":
            self.detector = Detector(args)
        elif args.mode == "question":
            self.questioner = Questioner(args)
        elif args.mode == "answer":
            self.answerer = Answerer(args)
            self.claim_generator = ClaimGenerator(args)
        
        print("Finish loading models.")

    
    def correct(self, sample: Dict):
        '''
        sample is Dict containing at least two fields:
            'input_desc': A passage that contains a description of the image.
            'input_img': Path to a local image 
        '''

        if self.args.mode == "entity_extract":
            # sample = self.preprocessor.generate_sentences(sample)
            sample = self.entity_extractor.extract_entity(sample)
        elif self.args.mode == "detecte":
            sample = self.detector.detect_objects(sample)
        elif self.args.mode == "question":
            sample = self.questioner.generate_questions(sample)
        elif self.args.mode == "answer":
            sample = self.answerer.generate_single_answers(sample)
            sample = self.claim_generator.generate_single_claim(sample)
            sample = self.answerer.generate_multiple_answers(sample)
            sample = self.claim_generator.generate_multiple_claim(sample)
        
        return sample

    def batch_correct(self, samples: List[Dict]):

        return [self.correct(sample) for sample in tqdm(samples, total=len(samples))]

def get_responses(datas, ids, output_file, model_args):
    if len(ids) != len(datas):
    # if len(ids) != 5:
        corrector = Corrector(model_args)
        for j in tqdm(range(len(datas))):
        # for j in tqdm(range(5)):
            if j in ids:
                continue
            sample = datas[j]
            sample = corrector.correct(sample)
            datas[j] = sample

            with open(output_file, "a+", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False)+'\n')

if __name__ == "__main__":
    args = get_args()
    args_dict = {
        'api_key': "8888",
        # 'api_base': args.api_base if args.api_base else "https://api.openai.com/v1",
        'api_base': "https://openkey.cloud/v1",
        'val_model_path': "/data/LLM/LLMs/Salesforce/blip2-flan-t5-xxl",
        'qa2c_model_path': "/data/LLM/LLMs/khhuang/zerofec-qa2claim-t5-base",
        'detector_config': "/home/projects/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        'detector_model_path': "/mnt/sda/models/groundingdino",
        'cache_dir': "./cache_dir",
}


    model_args = SimpleNamespace(**args_dict)
    model_args.device = args.device
    model_args.mode = args.mode
    datas = load_data(args)
    ids, output_file = process_result(args)
    setup_seeds(args.seed)
    get_responses(datas, ids, output_file, model_args)
