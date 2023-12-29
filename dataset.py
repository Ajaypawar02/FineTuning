
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import AdamW
from args import args

# class_mapping = {'scientific': 0, 'medical': 1, 'general': 2}
# corrected_df['target'] = corrected_df['context'].map(lambda x: class_mapping.get(x)) 
class_mapping = {
    'scientific': 0,
    'other': 12,
    'mathematics': 1,
    'medical': 2,
    'legal': 3,
    'technical': 4,
    'history': 5,
    'artistic': 6,
    'social': 7,
    'philosophical': 8,
    'political': 9,
    'health': 10,
    'biology': 11
}





class Data_class(Dataset):
    def __init__(self, df,args, inference_only=False):
        super().__init__()
        
        self.df = df      
        self.df['target'] = self.df['context'].map(lambda x: class_mapping.get(x)) 
        self.inference_only = inference_only
        self.text = self.df.query_eng.tolist()
        # print(self.text)
        
        if not self.inference_only:
            self.target = torch.tensor(df.target.values, dtype=torch.float)        
    
        self.encoded = args.tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = args.MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
#         print(self.encoded)

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index): 
        try:
            input_ids = torch.tensor(self.encoded['input_ids'][index])
            attention_mask = torch.tensor(self.encoded['attention_mask'][index])
        except:
            print(index)
        
        
        if self.inference_only:
            return {
                "input_ids" : input_ids, 
                "attention_mask" : attention_mask, 
            }           
        else:
            target = self.target[index]
            return {
                "input_ids" : input_ids, 
                "attention_mask" : attention_mask, 
                "target" : torch.tensor(target, dtype = torch.long)
            }


if __name__ == "__main__":
    file_path = "/home/ajay/SchoolHack/FineTuning/data/train_folds.csv"
    try:
        corrected_df = pd.read_csv(file_path)
        print(corrected_df["context"].value_counts())
        # print(corrected_df['query_eng'].isna().sum())
        corrected_df = corrected_df.dropna()
        corrected_df.reset_index(drop=True, inplace=True)
    except pd.errors.ParserError:
        # Handle a possible buffer overflow by reading the file in chunks
        chunk_size = 10000  # Adjust the chunk size based on your file size and memory constraints
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        corrected_df = pd.concat(chunks, ignore_index=True)
        corrected_df = corrected_df.dropna()
        corrected_df.reset_index(drop=True, inplace=True)
        print(corrected_df['query_eng'].isna().sum())
        print(corrected_df)

    # chunk_size = 10000  # Adjust the chunk size based on your file size and memory constraints
    # chunks = pd.read_csv(file_path, chunksize=chunk_size)
    # corrected_df = pd.concat(chunks, ignore_index=True)
    # print(corrected_df['query_eng'].isna().sum())
    # print(corrected_df)
    temp = Data_class(corrected_df, args)
    ans = temp.__getitem__(3)
    ids, mask, target = ans["input_ids"], ans["attention_mask"], ans["target"]
    print(ids.shape, mask.shape)