import os
#os.environ["CUDA_VISIBLE_DEVICES"]="6"
from typing import Dict
import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NUM_SECONDS_TO_SLEEP = 0.5



def format_inputs(question: str, answer: str):
    return f"{answer} \\n {question}"

def get_claim(tokenizer, model, question, answer):
    input_text = format_inputs(question, answer)
    input_ids = tokenizer(input_text, return_tensors="pt", padding='longest', truncation=True, max_length=512).input_ids.to(model.device)
    
    generated_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    ans = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return ans

def get_single_claim_string(sample):
    all_claim = sample['single_claim']
    specific_claims = ""
    count_claims = ""
    overal_claims = ""
    for entity in sample['entity_info']:
        count_claims += all_claim[entity]["counting_claim"]
        if all_claim[entity]["exist"]:
            
            for idx in range(len(all_claim[entity]["idx_claim_list"])):
                for j in range(len(all_claim[entity]["idx_claim_list"][idx])):
                    specific_claims += all_claim[entity]["idx_claim_list"][idx][j]
    if "others" in all_claim:
        overal_claims += "".join(all_claim["others"])

    claims = ""
    if count_claims != "":
        count_claims = "Counting:\n" + count_claims
        claims += count_claims + "\n"
    if specific_claims != "":
        specific_claims = "Single Object Level:\n" + specific_claims
        claims += specific_claims + "\n"
    
    sample['count_claims'] = count_claims
    sample['specific_claims'] = specific_claims 
    sample['overal_claims'] = overal_claims 

    sample['single_claim_string'] =  claims
    return sample

def get_multiple_claim_string(sample):
    all_claim = sample['multiple_claim']
    claims = ""
    if "multiple" in all_claim:
        claims += all_claim["multiple"]
    claims = sample['overal_claims'] + claims
    if claims != "":
        claims = "Multiple Objects Level:\n" + claims
    sample['multiple_claim_string'] = claims
    return sample


class ClaimGenerator:
    '''
        Input:
            dict: 
                'generated_questions': a list of 2-ele list, each [qs(str), involved entities(str)]
                                       each question is a list of 2 elements, [qs, entity]
                'generated_answers': An 1-d list of dict. Each dict in the list contains all the (qs, ans) tuple for each object instance.
                                {
                                    overall: [(qs, answer), ...]
                                    entity:  [
                                                [(qs, answer), ...]   (for instance 1 of this type of entity)
                                                    ...
                                             ]
                                }
        Output:
            dict:
                'claim': a global dict. merge 'generated_questions' and 'generated_answers' into sentence-level claims.
                            {
                                'specific':
                                    {
                                        entity 1: 2-d list. for each instance is a list: [claim1, claim2, ...]
                                        entity 2: 2-d list. for each instance is a list: [claim1, claim2, ...]   
                                    }
                                ...
                                'overall': 1-d list. 
                                'counting': 
                            }
    '''
    
    def __init__(self, args):
        self.args = args
        qa2c_model_path = args.qa2c_model_path
        self.model = AutoModelForSeq2SeqLM.from_pretrained(qa2c_model_path).to('cuda:0')
        self.tokenizer = AutoTokenizer.from_pretrained(qa2c_model_path)

    def generate_claim(self, sample: Dict):
        # claim from two parts. counting info and Q&A
        all_claim = {}
        
        # first part, Q&A
        generated_answers = sample['generated_answers']
        for answer_dict in generated_answers:
            for entity, answer_list in answer_dict.items():
                if entity == 'overall':
                    all_claim.setdefault('overall', [])
                    for qa_tuple in answer_list:
                        qs, ans = qa_tuple
                        clm = get_claim(self.tokenizer, self.model, qs, ans)
                        all_claim['overall'] += clm
                else:
                    all_claim.setdefault('specific', {}).setdefault(entity, [])
                    for idx, entity_answer_list in enumerate(answer_list):
                        if idx + 1 > len(all_claim['specific'][entity]):
                            all_claim['specific'][entity].append([])
                        for qa_tuple in entity_answer_list:
                            qs, ans = qa_tuple
                            clm = get_claim(self.tokenizer, self.model, qs, ans)
                            all_claim['specific'][entity][idx] += clm
                           
        # second part, counting info
        counting_claim = "Counting: \n"
        for entity, ent_info in sample['entity_info'].items():
            counting_claim_list = []
            ent_counts = ent_info['total_count']
            if ent_counts == 0:
                counting_claim += f"There is no {entity}.\n\n"
                continue
            else:
                counting_claim += f"There are {ent_counts} {entity}.\n"
                box_claim_list = []
                for idx, bbox in enumerate(ent_info['bbox']):
                    box_claim_list.append(f"{entity} {idx+1}: {bbox}.")
                counting_claim += '\n'.join(box_claim_list) + '\n\n'
                
        all_claim['counting'] = counting_claim
        sample['claim'] = all_claim     
        return sample
    
    def generate_single_claim(self, sample: Dict):
        # claim from two parts. counting info and Q&A
        all_claim = {}

        for entity, ent_info in sample['entity_info'].items():
            all_claim.setdefault(entity, {})
            counting_claim = ""
            ent_counts = ent_info['total_count']
            if ent_counts == 0:
                counting_claim += f"There is no {entity}.\n"
                all_claim[entity]["counting_claim"] = counting_claim
                all_claim[entity]["exist"] = False
                continue
            else:
                if ent_counts == 1:
                    counting_claim += f"There is {ent_counts} {entity}.\n"
                else:
                    counting_claim += f"There are {ent_counts} {entity}.\n"
                

                idx_claim_list = [[] for _ in range(len(ent_info['bbox']))]
                for idx, bbox in enumerate(ent_info['bbox']):
                    idx_claim_list[idx].append(f"{entity} {idx+1}: {bbox}\n")
                    # counting_claim += f"{entity} {idx+1}: {bbox}\n"
                    # idx_claim_list[idx] += f"{entity} {idx+1}:\nThe bounding box is {bbox}\n"
                all_claim[entity]["idx_claim_list"] = idx_claim_list
                all_claim[entity]["counting_claim"] = counting_claim
                all_claim[entity]["exist"] = True
        
        # first part, Q&A
        generated_answers = sample['generate_single_answers']
        
        for entity, answer_list in generated_answers.items():
            if entity == 'others':
                all_claim.setdefault('others', [])
                for qa_tuple in answer_list:
                    qs, ans = qa_tuple
                    clm = get_claim(self.tokenizer, self.model, qs, ans)
                    all_claim['others'].append(clm[0] + '\n')
            else:
                all_claim.setdefault(entity, [])
                for idx, entity_answer_list in enumerate(answer_list):
                    for qa_tuple in entity_answer_list:
                        qs, ans = qa_tuple
                        clm = get_claim(self.tokenizer, self.model, qs, ans)
                        all_claim[entity]["idx_claim_list"][idx].append(clm[0] + '\n')
                           
        sample['single_claim'] = all_claim   
        sample = get_single_claim_string(sample)  
        return sample

    
    def generate_multiple_claim(self, sample: Dict):
        # claim from two parts. counting info and Q&A
        all_claim = {}
        
        # first part, Q&A
        generated_answers = sample['generate_multiple_answers']
        # print("============")
        # print(generated_answers)
        # print("============")
        
        for entity, answer_list in generated_answers.items():
            if entity == 'multiple':
                all_claim.setdefault('multiple', "")
                for qa_tuple in answer_list:
                    qs, ans = qa_tuple
                    clm = get_claim(self.tokenizer, self.model, qs, ans)
                    # print("============")
                    # print(clm)
                    # print("============")
                    all_claim['multiple'] += clm[0] + '\n'

                           
        sample['multiple_claim'] = all_claim   
        sample = get_multiple_claim_string(sample)  
        return sample