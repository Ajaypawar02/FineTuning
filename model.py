import os
import math
import random
import time
from args import args
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import BertModel, BertTokenizer
from loss import focal_loss
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import model_selection
from tqdm import tqdm

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import gc
gc.enable()



class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start = 8, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        #print(weight_factor.shape)
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average



class SEN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(args.BERT_PATH)
        config.update({"output_hidden_states":True, 
                       "layer_norm_eps": 1e-7})                       
        self.layer_start = 9
        self.bert = AutoModel.from_pretrained(args.BERT_PATH, config=config)  

        self.attention = nn.Sequential(            
            nn.Linear(768, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        

        self.linear = nn.Linear(768, 13)
#         self.softmax = nn.Softmax(dim = -1)
        

    def forward(self, input_ids, attention_mask):
#         print(input_ids)
        outputs = self.bert(input_ids, attention_mask)
#         print(len(outputs))
        hidden_states = outputs[2]  # Assuming outputs[2] contains hidden_states

    # Use the hidden states from the last layer (or any specific layer)
        last_layer_hidden_states = hidden_states[-1]  # Last layer hidden states

        # Apply attention
        weights = self.attention(last_layer_hidden_states)

        # Compute the context vector
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)

        # Apply the final linear layer
        return self.linear(context_vector)
    
if __name__ == "__main__":
    model = SEN_Model()
    ids = torch.rand(256)
    mask = torch.rand(1)
    temp = model(ids.unsqueeze(0), mask.unsqueeze(0))
    print(temp.shape)