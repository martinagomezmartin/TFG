import OpenAttack
import pathlib
import gc
import sys
import ssl
import os  # Import the os module for setting environment variables
from datasets import Dataset
from pathlib import Path

from OpenAttack.tags import Tag
from OpenAttack.text_process.tokenizer import PunctTokenizer
from lambo.segmenter.lambo import Lambo
from victims.caching import VictimCache
from victims.bert import VictimBERT, readfromfile_generator
from victims.bilstm import VictimBiLSTM

from metrics.BODEGAScore import BODEGAScore
import time
import torch 

sys.path.append("/Users/martinagomez/Desktop/TFG/BODEGA/muss/")
from muss.simplify import ALLOWED_MODEL_NAMES, simplify_sentences, simplify_sentences_from_list

from transformers import AutoTokenizer
pretrained_model = "bert-base-uncased"

SEPARATOR_CHAR = '~'
SEPARATOR = ' ' + SEPARATOR_CHAR + ' '

# Set the environment variable TOKENIZERS_PARALLELISM to disable parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ssl._create_default_https_context = ssl._create_unverified_context

MAX_LEN = 512
MAX_ITER = 3

if len(sys.argv) > 0: 
    task = sys.argv[1]
    

class MyAttacker(OpenAttack.attackers.ClassificationAttacker):
    TAGS = {Tag("english", "lang"), Tag("get_pred", "victim")}
    
    def __init__(self):
        self.tokenizer = PunctTokenizer()
        self.lambo = Lambo.get('English')

    def attack(self, victim, input_, goal):
        x_new = self.simplify(input_)
        y_new = victim.get_pred([x_new])  
        if goal.check(x_new, y_new):
            return x_new
        return None
    
    def attack_loop(self, victim, input_, goal):
        x_new = input_ 
        for _ in range(MAX_ITER):
            x_new = self.simplify(x_new)  
            y_new = victim.get_pred([x_new])  
            if goal.check(x_new, y_new): 
                return x_new 
        return None 
    
    def attack_retry(self, victim, input_, goal):
        x_new = input_ 
        for _ in range(MAX_ITER):
            x_new = self.simplify(input)  
            y_new = victim.get_pred([x_new])  
            if goal.check(x_new, y_new): 
                return x_new 
        return None

    def simplify(self, document):
        sentences = []
        if task == "HN" or task == "RD":
            document = document.replace('\\n', '\n')
            document = document.split('\n')
            document = '.'.join(document)
        if task == "FC":
            split_text = document.split(SEPARATOR)
            document = split_text[0]
            part2 = split_text[1]
             
        document = self.lambo.segment(document)
        for turn in document.turns:
            for sentence in turn.sentences:
                sentences.append(sentence.text)
        simplified = simplify_sentences(sentences, model_name="muss_en_wikilarge_mined")
        if task == "FC":
            print(part2)
            part2_simple= simplify_sentences([part2], model_name="muss_en_wikilarge_mined")
            print(part2)
            simplified =  " ".join(simplified) + SEPARATOR + " ".join(part2_simple)
            print(simplified)
        else: 
            simplified = " ".join(simplified)
        return simplified

def trim(text, tokenizer):
    offsets = tokenizer(text, truncation=True, max_length=MAX_LEN + 10, return_offsets_mapping=True)['offset_mapping']
    limit = len(text)
    if len(offsets) > MAX_LEN:
        limit = offsets[512][1]
    return text[:limit]

def custom_generator(path,with_pairs):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    for line in open(path):
        parts = line.split('\t')
        label = int(parts[0])
        if with_pairs: 
            text1 = parts[2].strip().replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            text2 = parts[3].strip().replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')

            text1 = trim(text1, tokenizer)
            text2 = trim(text2, tokenizer)
            yield {'y': label, 'x': text1 + SEPARATOR + text2}
        else: 
            text = parts[2].strip().replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')   
            text = trim(text, tokenizer)
            yield {'y': label, 'x': text}

def main():
    victim_model = "BERT"
    targeted = True
    model_path_bodega = '/Users/martinagomez/Desktop/TFG/BODEGA/datasets/FC/BERT-512.pth'
    model_path_bodega = Path(model_path_bodega)
    #victim = OpenAttack.loadVictim("BERT.SST")
    
    if torch.cuda.is_available():
        victim_device = torch.device("cuda")

    else:
        victim_device = torch.device("cpu")
    
    print("Loading up victim model...")
    if victim_model == 'BERT':
        victim = VictimCache(model_path_bodega, VictimBERT(model_path_bodega, task, victim_device))
    elif victim_model == 'BiLSTM':
        victim = VictimCache(model_path_bodega, VictimBiLSTM(model_path_bodega, task, victim_device))

    
    attack = "Simplifiaction"
    with_pairs = (task == 'FC')
    

    
    RAW_FILE_NAME = 'raw_' + task + '_' + str(targeted) + '_' + attack + '_' + victim_model + 'test2'+'.tsv'
    out_dir = '/Users/martinagomez/Desktop/TFG/BODEGA/output/'
    raw_path = out_dir + RAW_FILE_NAME if out_dir else None

    
    FILE_NAME = 'results_' + task + '_' + str(targeted) + '_' + attack + '_' + victim_model +'test2'+ '.txt'
    dataset_path = "/Users/martinagomez/Desktop/TFG/BODEGA/datasets/"+task+"/attack.tsv"
    dataset = Dataset.from_generator(lambda: custom_generator(dataset_path, with_pairs))
    
    if targeted:
        dataset = [inst for inst in dataset if inst["y"] == 1 and victim.get_pred([inst["x"]])[0] == inst["y"]]
    print("The dataset data is:", dataset)

    attacker = MyAttacker()
    scorer = BODEGAScore(victim_device, task, align_sentences=True, semantic_scorer="BLEURT", raw_path = raw_path)
    
    attack_eval = OpenAttack.AttackEval(attacker, victim,language='english', metrics=[scorer])
    start = time.time()
    summary = attack_eval.eval(dataset, visualize=True, progress_bar=False)
    end = time.time()
    attack_time = end-start
    attacker = None

    # Evaluate
    start = time.time()
    score_success, score_semantic, score_character, score_BODEGA= scorer.compute()
    end = time.time()
    evaluate_time = end - start

    # Print results
    print("Subset size: " + str(len(dataset)))
    print("Success score: " + str(score_success))
    print("Semantic score: " + str(score_semantic))
    print("Character score: " + str(score_character))
    print("BODEGA score: " + str(score_BODEGA))
    print("Queries per example: " + str(summary['Avg. Victim Model Queries']))
    print("Total attack time: " + str(attack_time))
    print("Time per example: " + str((attack_time) / len(dataset)))
    print("Total evaluation time: " + str(evaluate_time))

    if out_dir:
        with open(out_dir + FILE_NAME, 'w') as f:
            f.write("Subset size: " + str(len(dataset)) + '\n')
            f.write("Success score: " + str(score_success) + '\n')
            f.write("Semantic score: " + str(score_semantic) + '\n')
            f.write("Character score: " + str(score_character) + '\n')
            f.write("BODEGA score: " + str(score_BODEGA) + '\n')
            f.write("Queries per example: " + str(summary['Avg. Victim Model Queries']) + '\n')
            f.write("Total attack time: " + str(end - start) + '\n')
            f.write("Time per example: " + str((end - start) / len(dataset)) + '\n')
            f.write("Total evaluation time: " + str(evaluate_time) + '\n')
            
    # Remove unused stuff
    del victim
    gc.collect()
    torch.cuda.empty_cache()
    
if __name__ == '__main__':
    main()
