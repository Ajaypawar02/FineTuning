
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import SEN_Model
from sklearn import metrics
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import BertModel, BertTokenizer
from args import args
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import model_selection
from tqdm import tqdm
from dataset import Data_class
from sklearn.model_selection import KFold
import warnings
from loss import focal_loss
from train_eval import train_fn, eval_fn
warnings.filterwarnings("ignore")
import gc
gc.enable()



from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def calculate_metrics(final_outputs, final_targets):
    # Convert probabilities to predicted classes
    predicted_classes = np.argmax(final_outputs, axis=1)
    true_classes = np.array(final_targets)

    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes, predicted_classes, average='weighted')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }

# Example usage
# final_outputs, final_targets = eval_fn(data_loader, model, device)
# metrics = calculate_metrics(final_outputs, final_targets)
# print(metrics)



def run(fold):
    df = pd.read_csv("./data/train_folds.csv")
#     df = df.iloc[:5000]
#     df_train, df_valid = model_selection.train_test_split(df, test_size = 0.1, random_state = 42)
    
    df_train = df[df["kfold"] != fold]
    df_valid = df[df["kfold"] == fold]
    
    
    df_train = df_train.reset_index(drop = True)
    df_valid = df_valid.reset_index(drop = True)
#     print(df_train)
    
    train_dataset = Data_class(df_train, args)
    valid_dataset = Data_class(df_valid, args)
    
    train_loader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = args.valid_batch_size)
    
    device = torch.device("cuda")
    model = SEN_Model()
    
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    
    optimizer_parameters = [
        {'params' : [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params' : [p for n, p in param_optimizer if  any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    num_training_steps = int(len(df_train)/args.train_batch_size)*args.epochs
    
    optimizer = AdamW(optimizer_parameters, lr = 3e-5)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = num_training_steps)
    
    model = model.to(device)
    # loss_list = []
    
    best_val = -1
    
    for epoch in range(10):
        loss_list = []
        loss = train_fn(train_loader, model, optimizer, device, scheduler)
        print(loss)
        # loss_list.append(loss)
        final_out, final_tar = eval_fn(valid_loader, model, device)
        print("================loss============", loss)
#         metrics = calculate_metrics(final_out, final_tar)

        metrics = calculate_metrics(final_out, final_tar)
        print(metrics)
#         print(metrics.classification_report(final_out, final_tar))
        
#         print("================validation===========")
    
#         f1_scores = f1_score(final_tar, final_out, average=None, labels=labels)
    
#         print("=============f1_scores======================", f1_scores)
        
#         f1_mean = f1_scores.mean()
#         print("=============f1_scores mean======================", f1_mean)

#         outputs = np.array(final_out) >= 0.5
        accuracy = metrics["f1_score"]
        loss_list.append((metrics, loss))
        file_path = f"./data/all_details_{epoch}.txt"
        dict_str = str(loss_list)
        with open(file_path, 'w') as f:
            f.write(dict_str)
        
        if best_val <= accuracy:
            print("======saving model============")
            best_val = accuracy
            model_path = "./data/" + r"modelsassa_{fold}_.pth".format(fold = fold)
            torch.save(model.state_dict(), model_path)
    file_path = "./data/all_details.txt"
    dict_str = str(loss_list)
    with open(file_path, 'w') as f:
        f.write(dict_str)
        
        
        
if __name__ == "__main__":
    # run(fold = 0)
    # run(fold = 1)
    # run(fold = 2)
    run(fold = 3)
    # run(fold = 4)
                