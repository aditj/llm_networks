import numpy as np
import random
import transformers
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import networkx as nx
import os
from datasets.llm_dataset import *
import argparse
import re
import ast
from utils.network import LLMNetwork
import json

if __name__ == "__main__":
    
    # test_dataset = None
    parser = argparse.ArgumentParser(description='Run LLM Network')
    parser.add_argument('--root_path', type=str, required=True, help='Root path to save results')
    parser.add_argument('--incentive', type=str, choices=['A', 'AR', 'CAR', 'Cr', 'Coq', 'Cot'], default='A')
    parser.add_argument('--ranking', type=str, choices=['+', "-", "&"], default="&")
    parser.add_argument('--connection', type=str, choices=['Powerlaw', "Chain", "Tree"], default="Powerlaw")
    args = parser.parse_args()
    
    
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    
    root_path = args.root_path
    if not os.path.exists(root_path):
        os.mkdir(root_path)
            
    with open('datasets/qa_dataset_fiction_v4.json', 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    # print(len(dataset))
    
    # incentive_list = ['A', 'AR', 'CAR', 'Cr']
    incentive_list = ['CAR', 'Cr']
    
    for incentive in incentive_list:
        save_path = os.path.join(root_path, incentive)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
                
        for i, data_item in enumerate(dataset[:5]):

            save_path_sub = os.path.join(save_path, str(i))
            if not os.path.exists(save_path_sub):
                os.mkdir(save_path_sub)
            
            network = LLMNetwork(num_agent=100, 
                                dataset="fiction",
                                data_item=data_item,
                                adj_path="utils/adj.npy",
                                save_path=save_path_sub,
                                model=model, tokenizer=tokenizer, args=args,
                                incentive = incentive,
                                ranking=args.ranking,
                                connection=args.connection
                                )
            
            network.multi_round_comm(10)
            
            
            ans_path = os.path.join(save_path_sub, "ans.json")
            all_answers = [agent.answer_list for agent in network.agent_list]
            with open(ans_path, "w") as file:
                json.dump(all_answers, file, indent=2)  # Optional: indent for readability
            
            # with open(ans_path, "w") as file:
            #     for agent in network.agent_list:
            #         file.write(str(agent.answer_list) + "\n")