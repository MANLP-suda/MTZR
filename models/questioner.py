from typing import Dict
from tqdm import tqdm
import openai
import time
import spacy

# Do not ask questions related to position or position relationship.
NUM_SECONDS_TO_SLEEP = 0.5
PROMPT_TEMPLATE_POSITION='''Given a sentence and some specified entities separated by "|", you need to ask questions about the positional relationships between these entities involved in the sentence to verify the factuality of the sentence.
Questions must only involve positional relationships between two objects.
Don't ask any questions unrelated to positional relationships.
Do not ask questions involving object counts or the existence of object.
You must specify the exact positional relationships between the objects.
The questions must be easily decided visually, without complex reasoning.
Avoid asking semantically similar questions. Avoid asking questions that only focus on scenes or places.
Avoid asking questions about uncertain or conjectural parts of the sentence, such as those expressed with "maybe," "likely," or similar terms.
It's not necessary to cover every specified entity. If there is no suitable question to ask, you should simply respond with 'None'.
When asking questions, avoid presuming the claims in the description are true beforehand. Focus solely on asking questions relevant to the information presented in the sentence.
Restrict your questions to common, specific, and concrete entities. The entities your questions involve should fall within the scope of the entities provided.
Output only one question in each line. For each line, present a question, followed by a single '&', and then list the involved entities, separated by "|" if there are multiple entities. Ensure the order of the entities matches their appearance in the question.

Examples:
Sentence:
The trash can is under the cup in the image.

Entities:
trash can|cup

Questions:
Is the trash can under the cup in the image?&trash can|cup

Sentence:
The white mouse is on the left of the keyboard.

Entities:
mouse|keyboard

Questions:
Is the white mouse on the left of the keyboard?&mouse|keyboard

Sentence:
{sent}

Entities:
{entity}

Questions:'''

PROMPT_TEMPLATE_EASY='''Given a sentence and some specified entities separated by "|", you need to ask relevant questions about these entities involved in the sentence to verify the factuality of the sentence.
Questions might involve basic attributes of entities, including colors, actions, etc.
Avoid asking questions that involve counting objects or confirming the existence of an object.
Do not ask questions about the positional relationship between objects.
When asking questions about attributes, try to ask simple questions that only involve one entity. 
The questions must be easily decided visually, without complex reasoning.
Avoid asking semantically similar questions. Avoid asking questions that only focus on scenes or places.
Avoid asking questions about uncertain or conjectural parts of the sentence, such as those expressed with "maybe," "likely," or similar terms.
It's not necessary to cover every specified entity. If there is no suitable question to ask, you should simply respond with 'None'.
When asking questions, avoid presuming the claims in the description are true beforehand. Focus solely on asking questions relevant to the information presented in the sentence.
Restrict your questions to common, specific, and concrete entities. The entities your questions involve should fall within the scope of the entities provided.
Output only one question in each line. For each line, present a question, followed by a single '&', and then list the involved entities, separated by "|" if there are multiple entities. Ensure the order of the entities matches their appearance in the question.

Examples:
Sentence:
There are one black dog and two white cats in the image.

Entities:
dog|cat

Questions:
What color is the cat?&cat
What color is the dog?&dog

Sentence:
The man is wearing a baseball cap and appears to be smoking.

Entities:
man

Questions:
What is the man wearing?&man
What is the man doing?&man

Sentence:
The image depicts a busy kitchen, with a man in a white apron. The man is standing in the middle of the kitchen.

Entities:
kitchen|man

Questions:
What does the man wear?&man

Sentence:
{sent}

Entities:
{entity}

Questions:'''

def remove_duplicates(res):
    qs_set = set()
    output = []
    for s in res:
        qs, ent = s
        if qs in qs_set:
            continue
        else:
            output.append(s)
            qs_set.add(qs)
    return output

def get_res_template(nlp, entity: str, sent: str, template, max_tokens: int=1024):
    content = template.format(sent=sent, entity=entity)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a language assistant that helps to ask questions about a sentence.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    res = response['choices'][0]['message']['content'].splitlines()
    res = [s.split('&') for s in res if s.lower() != 'none']
    
    return res


def get_res(nlp, entity: str, sent: str, max_tokens: int=1024):
    single_qs = get_res_template(nlp, entity, sent, PROMPT_TEMPLATE_EASY)
    position_qs = get_res_template(nlp, entity, sent, PROMPT_TEMPLATE_POSITION)
    res = single_qs + position_qs
    # res = position_qs
    entity_list = entity.split('|')
    
    res = [s for s in res if len(s)==2]
    res = remove_duplicates(res)
    res = [s for s in res if set(s[1].split('|')).issubset(set(entity_list)) ]

    return res

class Questioner:
    '''
        Input:
            For each splitted sentences:
                A sentence and list of existent objects. (only questions about existent objects)
        Output:
            For each splitted sentences:
                A list of 2-ele list: [[question, involved object type], [qs, obj], ...]         
    '''
    def __init__(self, args):
        
        openai.api_key = args.api_key
        openai.api_base = args.api_base
        self.args = args
    
        self.nlp = spacy.load("en_core_web_sm")
        
    def generate_questions(self, sample: Dict):
        sentences = sample['split_sents']
        global_entity_dict = sample['entity_info']
        global_entity_list = sample['entity_list']
        
        qs_list = []
        for ent_list, sent in zip(global_entity_list, sentences):
            exist_entity = [ent for ent in ent_list if ent in global_entity_dict and global_entity_dict[ent]['total_count'] > 0]
            
            # border case: no detection result for any entity. no question asked.
            if len(exist_entity)==0 :
                qs_list.append([])
                continue
            
            questions = get_res(self.nlp, '|'.join(exist_entity), sent)
            qs_list.append(questions)
        sample['generated_questions'] = qs_list

        single_qs_list = []
        multiple_qs_list = []
        for questions in qs_list:
            for qs, entity in questions:
                entity_list = entity.split('|')
                entity_list = [e.strip() for e in entity_list if e.strip()]
                if len(entity_list) == 1:
                    single_qs_list.append((qs, entity))
                else:
                    multiple_qs_list.append((qs, entity))

        sample['single_qs_list'] = single_qs_list
        sample['multiple_qs_list'] = multiple_qs_list
        
        return sample
    