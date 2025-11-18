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
from utils.agent import LLMAgent

class LLMNetwork:
    def __init__(self, 
                 num_agent=100, 
                 dataset="fiction",
                 data_item = None,
                 adj_path = None,
                 save_path=None,
                 model=None, tokenizer=None, args=None, 
                 incentive="A",
                 ranking="Random",
                 connection="Powerlaw"
                 ):
        
        ##################################################        
        self.num_agent = num_agent
        self.dataset = dataset
        self.data_item = data_item
        self.default_single_prompt = [
            {"role": "system", "content": "Default."},
            {"role": "user", "content": "Default."},
        ]
        self.default_single_prompt_two_round = [
            {"role": "system", "content": "Default."},
            {"role": "user", "content": "Default."},
            {"role": "assistant", "content": "Default."},
            {"role": "user", "content": "Default."},
        ]
        self.default_single_prompt_three_round = [
            {"role": "system", "content": "Default."},
            {"role": "user", "content": "Default."},
            {"role": "assistant", "content": "Default."},
            {"role": "user", "content": "Default."},
            {"role": "assistant", "content": "Default."},
            {"role": "user", "content": "Default."},
        ]
        self.default_single_prompt_four_round = [
            {"role": "system", "content": "Default."},
            {"role": "user", "content": "Default."},
            {"role": "assistant", "content": "Default."},
            {"role": "user", "content": "Default."},
            {"role": "assistant", "content": "Default."},
            {"role": "user", "content": "Default."},
            {"role": "assistant", "content": "Default."},
            {"role": "user", "content": "Default."},
        ]
        
        self.adj = np.load(adj_path)
        # self.adj = Non
        self.incentive = incentive
        self.adj_path = adj_path
        # self.adj = self.init_connection(ranking="&", connection="Power")
        # self.adj = self.init_connection(ranking, connection)
        
        self.agent_list = self.init_agent_list()
        
        ##################################################    
        
        self.args = args
        self.total_num_input_tokens = 0
        self.total_num_output_tokens = 0
        
        
        self.default_single_prompt = [
            {"role": "system", "content": "Default."},
            {"role": "user", "content": "Default."}
        ]
        
        self.agent_list = self.init_agent_list()
        self.save_path = save_path

            
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda",
            batch_size= 128, 
        )
        
    def create_adjacency_matrix(self, n: int, k: float, mode="degree-distri") -> np.ndarray:
        # Initialize an nxn matrix with zeros
        adjacency_matrix = np.zeros((n, n), dtype=int)
        
        if mode == "erdos-Gnp":
            # Iterate over the upper triangle of the matrix (excluding the diagonal)
            for i in range(n):
                for j in range(n):
                    if np.random.rand() < (k / 100):  # Convert k% to a probability
                        adjacency_matrix[i, j] = 1 
                        
        elif mode == "erdos-GnM":
            possible_edges = [(i, j) for i in range(n) for j in range(n)]
            M = int(k * n/100)
            selected_edges = random.sample(possible_edges, M)
            
            for i, j in selected_edges:
                adjacency_matrix[i, j] = 1 
            
        elif mode == "perfer-attach":
            
            m = int(k*n/100)
            # Generate the init adj matrix
            for i in range(1, m+1):
                adjacency_matrix[i, 0] = 1
                
            in_degrees = [m] + [1] * m
            for new_node in range(m+1, n):
                # Select m existing nodes as targets with probability proportional to their in-degree
                possible_targets = random.choices(range(new_node), weights=in_degrees, k=m)
                
                # Add directed edges from the new node to the selected targets
                for target in possible_targets:
                    adjacency_matrix[new_node, target] = 1  # Edge from new_node -> target

                # Update in-degree list
                in_degrees.append(0)  # New node starts with 0 in-degree
                for target in possible_targets:
                    in_degrees[target] += 1  # Increase in-degree of target nodes
                    
            pass
        
        elif mode == "degree-distri":
            r = int(k*n/100)
            
            alpha = r/10
            size = self.num_agent
            
            int_arr = (np.random.pareto(alpha, size)*5+1).astype(int)
            
            for i in range(size):
                possible_edges = [(i, j) for j in range(n) if j != i]
                selected_edges = random.sample(possible_edges, min(int_arr[i].item(), int(n/2)))
                # print(len(selected_edges))
                
                for a, b in selected_edges:
                    adjacency_matrix[a, b] = 1
        else:
            pass
        # Ensure every agent has at least one attached neighbor
        for i in range(n):
            if not np.any(adjacency_matrix[i]):  # If row i has only zeros (no neighbors)
                possible_neighbors = list(set(range(n)) - {i})  # All other agents
                neighbor = np.random.choice(possible_neighbors)  # Randomly select a neighbor
                adjacency_matrix[i, neighbor] = 1
                adjacency_matrix[neighbor, i] = 1  # Ensure symmetry

        return adjacency_matrix
    
    def init_connection(self, ranking, connection):
        if connection == "Chain":
            if ranking == "+":
                pass
            elif ranking == "-":
                pass
            else:
                pass
                
        elif connection == "Tree":
            if ranking == "+":
                pass
            elif ranking == "-":
                pass
            else:
                pass
        
        else:
            if ranking == "+":
                pass
            elif ranking == "-":
                pass
            else:
                self.adj = np.load(self.adj_path)
        return
    
    def count_tokens(self, prompt):
    # Calculate the total number of tokens of a prompt in the format of list or string
        if isinstance(prompt, list):
            formatted_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
        else:
            formatted_text = prompt
            
        tokenized_output = self.tokenizer(formatted_text, return_tensors="pt")
        num_tokens = tokenized_output["input_ids"].shape[1]
        
        return num_tokens
    
    def get_init_prompt(self, text_question, text_narrative, text_choices):
        
        if self.dataset == "fiction":
            
            starting_prompt = "You are a question answering agent which receives part of the text. You need to choose the most suitable answer among the choices provided to you. "
            narrative_prompt = (
                "You are giving this part of the text: \n"
                f"{text_narrative}"
            )         
            
        elif self.dataset == "event":
            starting_prompt = "You are a question answering agent which receives a certain multi-media narrative type of an event in the world. You need to choose the most suitable answer among the choices provided to you. "
            narrative_prompt = (
                "You are giving this narrative of the event: \n"
                f"{text_narrative}"
            )
            
        elif self.dataset == "cutoff":
            starting_prompt = "You are a question answering agent which receives a question regarding the fact of the world. You need to choose the most suitable answer among the choices provided to you. "
            narrative_prompt = ""

        else:
            raise ValueError(f"Unknown dataset type: {self.dataset}")
        
        # One round prompt
        
        if self.incentive == "A":
        
            question_prompt = (
                "You will be given a question and multiple-choice options.\n\n"
                f"Question:\n{text_question}\n\n"
                f"Choices:\n{text_choices}\n\n"
                "Instructions:\n"
                "- Answer with the letter of the correct choice only (e.g., A, B, C).\n"
                "- Base your answer strictly on the provided text.\n"
                "- Do not use outside knowledge. Using external knowledge will be considered incorrect.\n"
                "- If the text does not provide enough information, answer 'I don't know'. Be honest to your answer.\n"
            )
            
            final_init_prompt = copy.deepcopy(self.default_single_prompt)
            final_init_prompt[0]["content"] = starting_prompt + "\n" + narrative_prompt
            final_init_prompt[1]["content"] = question_prompt
        
        elif self.incentive == "AR":
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "You have to give a reason in about 20 words."
            )
            
            question_prompt_final = (
                "Give the final anwer, you only have to answer the choice letter. "
            )
            
            final_init_prompt = copy.deepcopy(self.default_single_prompt)
            final_init_prompt[0]["content"] = starting_prompt + "\n" + narrative_prompt
            final_init_prompt[1]["content"] = question_prompt
            final_init_prompt.append({"role": "user", "content": question_prompt_final})
            
        # Two rounds prompt
        
        elif self.incentive == "CAR":
            
            question_prompt = (
                "This is the question you will eventually answer: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "Before answering, critique the clarity, fairness, or ambiguity of the question and choices in about 30 words."
            )
            question_prompt_1 = (
                "You have already critiqued the question. Now proceed to answer it.\n"
                "You have to answer the choice letter first, enclosed by <>, then give a reason in about 20 words."
            )
            
            question_prompt_final = (
                "Give the final anwer, you only have to answer the choice letter. "
            )
            
            final_init_prompt = copy.deepcopy(self.default_single_prompt)
            final_init_prompt[0]["content"] = starting_prompt + "\n" + narrative_prompt
            final_init_prompt[1]["content"] = question_prompt
            final_init_prompt.append({"role": "user", "content": question_prompt_1})
            final_init_prompt.append({"role": "user", "content": question_prompt_final})
        
        elif self.incentive == "Cr":
            
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "Examine answer choices and compare them, explaining which is stronger and why, in about 30 words in total."
            )
            question_prompt_1 = (
                "Based on your earlier contrastive reasoning, select the best answer.\n"
                "You have to answer the choice letter first, enclosed by <>, then give a reason in about 20 words."
            )
            question_prompt_final = (
                "Give the final anwer, you only have to answer the choice letter. "
            )
            final_init_prompt = copy.deepcopy(self.default_single_prompt)
            final_init_prompt[0]["content"] = starting_prompt + "\n" + narrative_prompt
            final_init_prompt[1]["content"] = question_prompt
            final_init_prompt.append({"role": "user", "content": question_prompt_1})
            final_init_prompt.append({"role": "user", "content": question_prompt_final})
            
        # Three rounds prompt
        
        elif self.incentive == "Coq":
            
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "Solve the problem provided to you based on the paragraph provided to you part by part. Try to answer the question based on the first one-third of the orginal text that are given to you."
            )
            question_prompt_1 = (
                "Try to answer the question based on the second one-third of the orginal text that are given to you."
            )
            question_prompt_2 = (
                "Try to answer the question based on the second one-third of the orginal text that are given to you. \n"
                "Then give your final answer. You have to answer the choice letter first, enclosed by <>, then give a reason in about 20 words."
            )
            question_prompt_final = (
                "Give the final anwer, you only have to answer the choice letter. "
            )
            final_init_prompt = copy.deepcopy(self.default_single_prompt)
            final_init_prompt[0]["content"] = starting_prompt + "\n" + narrative_prompt
            final_init_prompt[1]["content"] = question_prompt
            final_init_prompt.append({"role": "user", "content": question_prompt_1})
            final_init_prompt.append({"role": "user", "content": question_prompt_2})
            final_init_prompt.append({"role": "user", "content": question_prompt_final})
            
        elif self.incentive == "Cot":
            
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "Solve the problem provided to you based on chain of thoughts in three steps. Start reasoning step-by-step. Write your first thought in about 20 words."
            )
            question_prompt_1 = (
                "Continue your chain of thought. Write the next logical step in 20-30 words."
            )
            question_prompt_2 = (
                "Finish your last step of chain of thought. And conclude your final answer.\n"
                "You have to answer the choice letter first, enclosed by <>, then give a reason in about 20 words."
            )
            question_prompt_final = (
                "Give the final anwer, you only have to answer the choice letter. "
            )
            final_init_prompt = copy.deepcopy(self.default_single_prompt)
            final_init_prompt[0]["content"] = starting_prompt + "\n" + narrative_prompt
            final_init_prompt[1]["content"] = question_prompt
            final_init_prompt.append({"role": "user", "content": question_prompt_1})
            final_init_prompt.append({"role": "user", "content": question_prompt_2})
            final_init_prompt.append({"role": "user", "content": question_prompt_final})
            
        else:
            raise ValueError(f"Unknown incentive type: {self.incentive}")
            
        return final_init_prompt
    
    def init_agent_list(self):
        
        def get_key(index):
            return "narrative_" + str(index)
        
        def get_choices(list_choices):
            
            choice_prompt = ""
            list_index = ["A", "B", "C", "D", "E", "F", "G"]
            
            for i, choice in enumerate(list_choices):
                choice_prompt += "Choice " + list_index[i] + " is: " + choice + "\n"
                
            return choice_prompt 
                
        agent_flag_list = [i for i in range(5) for _ in range(int(self.num_agent/5))]
        
        # print(len(agent_flag_list))
        random.shuffle(agent_flag_list)
        
        print(agent_flag_list)
        
        init_agent_list = []
        
        if self.dataset == "fiction" or self.dataset == "event":
            
            for i, flag in enumerate(agent_flag_list):
                
                init_prompt = ""
                narrative_key = get_key(flag)
                text_question = self.data_item["hallucination_questions"]
                text_narrative = self.data_item[narrative_key]
                text_choices = get_choices([self.data_item["answer_0"], self.data_item["answer_1"], self.data_item["answer_2"]])
                
                init_prompt = self.get_init_prompt(text_question, text_narrative, text_choices)
                
                init_agent_list.append(LLMAgent(init_prompt=init_prompt, agent_index=i))
                
        elif self.dataset == "cutoff":
            for i, flag in enumerate(agent_flag_list):
                
                init_prompt = ""
                narrative_key = get_key(flag)
                text_question = self.data_item["hallucination_questions"]
                text_narrative = ""
                text_choices = get_choices([self.data_item["answer_0"], self.data_item["answer_1"], self.data_item["answer_2"]])
                
                init_prompt = self.get_init_prompt(text_question, text_narrative, text_choices)
                
                init_agent_list.append(LLMAgent(init_prompt=init_prompt, agent_index=i))
            pass
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset}")
        
        # return a list of agent
        # Each agent is initilzed with the memory in the form of a list
        # The first item of the list is the system prompt is the starting prompt to introduce the duty of llm and the prior text given to them. (No prior text for cutoff dataset)
        # The second item of the list is the question and choice prompt giving to them. Incentives prompt can also be included in this sense.
        return init_agent_list 
       
    def get_attached_agents(self, agent_index: int) -> list:
        return list(np.where(self.adj[agent_index] == 1)[0])
    
    def init_ask(self):
        
        if self.incentive == "A":
                
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[-1]
                batched_prompts.append(single_prompt)
            outputs = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=4, 
                        return_full_text=False
                        )
                )
            self.update_agents(batched_prompts, outputs) # Add answer
            
        elif self.incentive == "AR":
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[1]
                batched_prompts.append(single_prompt)
            outputs1 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs1.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            
            # print(outputs1)
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_two_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[1]
                single_prompt[2]["content"] =  outputs1[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                batched_prompts.append(single_prompt)
            outputs2 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs2.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=4, 
                        return_full_text=False
                        )
                )
            # print(outputs2)
            
            outputs_total = []
            for i in range(self.num_agent):
                output_total = ["Choice: [" + outputs2[i][0]["generated_text"] + "]. " + outputs1[i][0]["generated_text"]]
                
                outputs_total.append([{"generated_text": output_total}])
            self.update_agents(batched_prompts, outputs_total)
        
        elif self.incentive == "CAR" or self.incentive == "Cr":
            
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[1]
                batched_prompts.append(single_prompt)
            outputs0 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs0.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_two_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[1]
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                batched_prompts.append(single_prompt)
            outputs1 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs1.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_three_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[1]
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                single_prompt[4]["content"] =  outputs1[i]
                single_prompt[5]["content"] =  self.agent_list[i].memory[3]
                batched_prompts.append(single_prompt)
            outputs2 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs2.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=4, 
                        return_full_text=False
                        )
                )
                
            outputs_total = []
            for i in range(self.num_agent):
                output_total = "Choice: [" + outputs2[i][0]["generated_text"] + "]. " + outputs1[i][0]["generated_text"]
                
                # print("###############################################")
                # print(output_total)
                # print("###############################################")
                
                outputs_total.append([{"generated_text": output_total}])
            self.update_agents(batched_prompts, outputs_total)
                
        elif self.incentive == "Coq" or self.incentive == "Cot":
            
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[1]
                batched_prompts.append(single_prompt)
            outputs0 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs0.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_two_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[1]
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                batched_prompts.append(single_prompt)
            outputs1 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs1.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_three_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[1]
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                single_prompt[4]["content"] =  outputs1[i]
                single_prompt[5]["content"] =  self.agent_list[i].memory[3]
                batched_prompts.append(single_prompt)
            outputs2 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs2.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_four_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.agent_list[i].memory[1]
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                single_prompt[4]["content"] =  outputs1[i]
                single_prompt[5]["content"] =  self.agent_list[i].memory[3]
                single_prompt[6]["content"] =  outputs2[i]
                single_prompt[7]["content"] =  self.agent_list[i].memory[4]
                batched_prompts.append(single_prompt)
            outputs3 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs3.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=4, 
                        return_full_text=False
                        )
                )
                
            outputs_total = []
            for i in range(self.num_agent):
                output_total = ["Choice: [" + outputs3[i][0]["generated_text"] + "]. " + outputs2[i][0]["generated_text"]]
                outputs_total.append([{"generated_text": output_total}])
            self.update_agents(batched_prompts, outputs_total)
                
        else:
            raise ValueError(f"Unknown incentive type: {self.incentive}")
        
        
        return
    
    def one_round_comm(self):
                
        if self.incentive == "A":
                
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                batched_prompts.append(single_prompt)
                
            
            # print(batched_prompts[2])
            outputs = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=4, 
                        return_full_text=False
                        )
                )
            self.update_agents(batched_prompts, outputs) # Add answer
            
        elif self.incentive == "AR":
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                batched_prompts.append(single_prompt)
            outputs1 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs1.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            
            # print(outputs1)
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_two_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                single_prompt[2]["content"] =  outputs1[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                batched_prompts.append(single_prompt)
            outputs2 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs2.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=4, 
                        return_full_text=False
                        )
                )
            # print(outputs2)
            
            outputs_total = []
            for i in range(self.num_agent):
                output_total = ["Choice: [" + outputs2[i][0]["generated_text"] + "]. " + outputs1[i][0]["generated_text"]]
                
                outputs_total.append([{"generated_text": output_total}])
            self.update_agents(batched_prompts, outputs_total)
        
        elif self.incentive == "CAR" or self.incentive == "Cr":
            
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                batched_prompts.append(single_prompt)
            outputs0 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs0.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_two_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                batched_prompts.append(single_prompt)
            outputs1 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs1.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_three_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                single_prompt[4]["content"] =  outputs1[i]
                single_prompt[5]["content"] =  self.agent_list[i].memory[3]
                batched_prompts.append(single_prompt)
            outputs2 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs2.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=4, 
                        return_full_text=False
                        )
                )
                
            outputs_total = []
            for i in range(self.num_agent):
                output_total = "Choice: [" + outputs2[i][0]["generated_text"] + "]. " + outputs1[i][0]["generated_text"]
                outputs_total.append([{"generated_text": output_total}])
            self.update_agents(batched_prompts, outputs_total)
                
        elif self.incentive == "Coq" or self.incentive == "Cot":
            
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                batched_prompts.append(single_prompt)
            outputs0 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs0.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_two_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                batched_prompts.append(single_prompt)
            outputs1 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs1.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_three_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                single_prompt[4]["content"] =  outputs1[i]
                single_prompt[5]["content"] =  self.agent_list[i].memory[3]
                batched_prompts.append(single_prompt)
            outputs2 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs2.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=35, 
                        return_full_text=False
                        )
                )
            batched_prompts = []
            for i in range(len(self.agent_list)):
                single_prompt = copy.deepcopy(self.default_single_prompt_four_round)
                single_prompt[0]["content"] =  self.agent_list[i].memory[0]
                single_prompt[1]["content"] =  self.create_center_agent_prompt(i)
                single_prompt[2]["content"] =  outputs0[i]
                single_prompt[3]["content"] =  self.agent_list[i].memory[2]
                single_prompt[4]["content"] =  outputs1[i]
                single_prompt[5]["content"] =  self.agent_list[i].memory[3]
                single_prompt[6]["content"] =  outputs2[i]
                single_prompt[7]["content"] =  self.agent_list[i].memory[4]
                batched_prompts.append(single_prompt)
            outputs3 = []
            batch_size = 10
            for i in range(0, len(batched_prompts), batch_size):
                batch = batched_prompts[i:i + batch_size]
                outputs3.extend(
                    self.pipeline(
                        batch, 
                        max_new_tokens=4, 
                        return_full_text=False
                        )
                )
                
            outputs_total = []
            for i in range(self.num_agent):
                output_total = ["Choice: [" + outputs3[i][0]["generated_text"] + "]. " + outputs2[i][0]["generated_text"]]
                outputs_total.append([{"generated_text": output_total}])
            self.update_agents(batched_prompts, outputs_total)
        else:
            raise ValueError(f"Unknown incentive type: {self.incentive}")
        
        
        return 
    
    def create_center_agent_prompt(self, center_agent_index):
        
        def get_choices(list_choices):
            
            choice_prompt = ""
            list_index = ["A", "B", "C", "D", "E", "F", "G"]
            
            for i, choice in enumerate(list_choices):
                choice_prompt += "Choice " + list_index[i] + " is: " + choice + "\n"
                
            return choice_prompt 
        
        query_agents_index = self.get_attached_agents(center_agent_index)
        
        text_question = self.data_item["hallucination_questions"]
        text_choices = get_choices([self.data_item["answer_0"], self.data_item["answer_1"], self.data_item["answer_2"]])
        
        if self.incentive == "A":
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "You only have to answer the choice letter."
            )
            
            neighbor_prompt = "Your neighbouring agents provided the following answers: "
            for agent_index in random.sample(query_agents_index, min(10, len(query_agents_index))):
                neighbor_prompt += "Agent with index " + str(agent_index) + " provides an answer: " + self.agent_list[agent_index].answer_list[-1] + "\n"
            prev_answer_prompt = "Your previous answer to this question was " + self.agent_list[center_agent_index].answer_list[-1] + ". " + "\n"
            
            choice_prompt = (
                "Ensure you only answer the choice letter."
            )
        
        elif self.incentive == "AR":
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "You have to answer the choice letter first, enclosed by <>, then give a reason in about 20 words."
            )
            
            neighbor_prompt = "Your neighbouring agents provided the following answers: "
            for agent_index in random.sample(query_agents_index, min(10, len(query_agents_index))):
                
                # print("###################")
                # print(self.agent_list[agent_index].answer_list[-1])
                # print("###################")
                # print(type(self.agent_list[agent_index].answer_list[-1]))
                # print("###################")
                neighbor_prompt += "Agent with index " + str(agent_index) + " provides an answer: " + self.agent_list[agent_index].answer_list[-1] + "\n"
            prev_answer_prompt = "Your previous answer to this question was " + self.agent_list[center_agent_index].answer_list[-1] + ". " + "\n"
            
            choice_prompt = (
                "You have to answer the choice letter first, enclosed by <>, then give a reason in about 20 words."
            )
        
        elif self.incentive == "CAR":
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "Before answering, critique the clarity, fairness, or ambiguity of the question and choices in about 30 words."
            )
            
            neighbor_prompt = "Your neighbouring agents provided the following answers: "
            for agent_index in random.sample(query_agents_index, min(10, len(query_agents_index))):
                neighbor_prompt += "Agent with index " + str(agent_index) + " provides an answer: " + self.agent_list[agent_index].answer_list[-1] + "\n"
            prev_answer_prompt = "Your previous answer to this question was " + self.agent_list[center_agent_index].answer_list[-1] + ". " + "\n"
            
            choice_prompt = (
                "Before answering, critique the clarity, fairness, or ambiguity of the question and choices in about 30 words."
            )
        
        elif self.incentive == "Cr":
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "Examine answer choices and compare them, explaining which is stronger and why, in about 40 words in total."
            )
            
            neighbor_prompt = "Your neighbouring agents provided the following answers: "
            for agent_index in random.sample(query_agents_index, min(10, len(query_agents_index))):
                neighbor_prompt += "Agent with index " + str(agent_index) + " provides an answer: " + self.agent_list[agent_index].answer_list[-1] + "\n"
            prev_answer_prompt = "Your previous answer to this question was " + self.agent_list[center_agent_index].answer_list[-1] + ". " + "\n"
            
            choice_prompt = (
                "Examine answer choices and compare them, explaining which is stronger and why, in about 40 words in total."
            )
        
        elif self.incentive == "Coq":
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "Solve the problem provided to you based on the paragraph provided to you part by part. Try to answer the question based on the first one-third of the orginal text that are given to you."
            )
            
            neighbor_prompt = "Your neighbouring agents provided the following answers: "
            for agent_index in random.sample(query_agents_index, min(10, len(query_agents_index))):
                neighbor_prompt += "Agent with index " + str(agent_index) + " provides an answer: " + self.agent_list[agent_index].answer_list[-1] + "\n"
            prev_answer_prompt = "Your previous answer to this question was " + self.agent_list[center_agent_index].answer_list[-1] + ". " + "\n"
            
            choice_prompt = (
                "Solve the problem provided to you based on the paragraph provided to you part by part. Try to answer the question based on the first one-third of the orginal text that are given to you."
            )
        
        elif self.incentive == "Cot":
            question_prompt = (
                "This is the question: \n"
                f"{text_question}"
                "These are the choices: \n"
                f"{text_choices}"
                "Solve the problem provided to you based on chain of thoughts in three steps. Start reasoning step-by-step. Write your first thought in about 20 words."
            )
            
            neighbor_prompt = "Your neighbouring agents provided the following answers: "
            for agent_index in random.sample(query_agents_index, min(10, len(query_agents_index))):
                neighbor_prompt += "Agent with index " + str(agent_index) + " provides an answer: " + self.agent_list[agent_index].answer_list[-1] + "\n"
            prev_answer_prompt = "Your previous answer to this question was " + self.agent_list[center_agent_index].answer_list[-1] + ". " + "\n"
            
            choice_prompt = (
                "Solve the problem provided to you based on chain of thoughts in three steps. Start reasoning step-by-step. Write your first thought in about 20 words."
            )
        
        else:
            raise ValueError(f"Unknown incentive type: {self.incentive}")
        
        return question_prompt + neighbor_prompt + prev_answer_prompt + choice_prompt
    
    def multi_round_comm(self, num_round):
        self.init_ask()
        for i in tqdm(range(num_round)):
            self.one_round_comm()
        return
    
    def update_agents(self, batched_prompts, outputs):
        
        for i, output in enumerate(outputs):
            agent = self.agent_list[i]
            # agent.clear_memory()
            # print(output[0]['generated_text'])
            agent.add_answer(output[0]['generated_text'])
            self.agent_list[i] = agent
            
        return
