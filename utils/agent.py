import numpy as np
import random
import transformers
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import networkx as nx

class LLMAgent:
    def __init__(self, init_prompt="", agent_index=None):
        # self.memory = []  # Memory to store interaction history
        self.memory = init_prompt # Memory to store interaction history
        
        # self.init_prompt = init_prompt
        # self.add_to_memory(init_prompt)
        self.answer_list = []
        self.tidy_answer_list = []
        self.agent_index = agent_index
    

    def add_to_memory(self, message):
        if isinstance(message, dict) and 'role' in message and 'content' in message:
            self.memory.append(message)
        else:
            raise ValueError("Message must be a dictionary with 'role' and 'content'.")

    # def clear_memory(self):
    #     init_memory = self.memory[:1]
    #     self.memory = []
    #     self.memory.append(init_memory)
        
    def add_ask(self, question):
        self.add_to_memory({"role": "user", "content": question})
        
    def add_answer(self, model_response):
        # input is just a string
        # append the last answer to memory list 
        # append the answer to the answer list
        self.add_to_memory({"role": "assistant", "content": model_response})
        
        if type(model_response) == list:
            self.answer_list.append(model_response[0])
        else:
            self.answer_list.append(model_response)
        return model_response
    