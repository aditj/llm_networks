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
    parser.add_argument('--network_size', type=int, default=100)
    parser.add_argument('--model_path', type=str, default="/meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--adj_path', type=str, default="utils/adj_100_gnm.npy")
    parser.add_argument('--data_path', type=str, default="qa_dataset_event_v3.json")
    
    args = parser.parse_args()
    
    model_path = args.model_path
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
            
    with open('qa_dataset_fiction_v6.json', 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    # print(len(dataset))
    for i, data_item in enumerate(dataset):
        
        save_path = os.path.join(root_path, str(i))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        
        print(data_item["hallucination_questions"])
        
        network = LLMNetwork(num_agent=args.network_size, 
                            dataset=args.data_path.split("_")[2],
                            data_item=data_item,
                            adj_path=args.adj_path,
                            save_path=save_path,
                            model=model, tokenizer=tokenizer, args=args,
                            incentive = args.incentive,
                            ranking=args.ranking,
                            connection=args.connection
                            )
        
        network.multi_round_comm(10)
        
        
        ans_path = os.path.join(save_path, "ans.txt")
        with open(ans_path, "w") as file:
            for agent in network.agent_list:
                file.write(str(agent.answer_list) + "\n")
        
        ans_path = os.path.join(save_path, "ans.json")

        with open(ans_path, "w") as file:
            # Collect all answer lists into a single structure
            all_answers = [agent.answer_list for agent in network.agent_list]
            # Write as JSON
            json.dump(all_answers, file, indent=4)
