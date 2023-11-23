import os  
import numpy as np 
import collections
from collections import defaultdict
from tqdm import tqdm
import torch 
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import (AdamW, RobertaTokenizer,
                          get_linear_schedule_with_warmup)


def read_file(file_dir="conll", dataset_name="train"):
        # text_file = os.path.join(file_dir, f"{dataset_name}_text.txt")
        text_file = './data/conll/train_text.txt'
        f_text = open(text_file, encoding='utf-8')
        text_contents = f_text.readlines()
        # label_file = os.path.join(file_dir, f"{dataset_name}_label.txt")
        label_file = './data/conll/train_label_true.txt'
        f_label = open(label_file)
        label_contents = f_label.readlines()
        sentences = []
        labels = []
        for text_line, label_line in zip(text_contents, label_contents):
            sentence = text_line.strip().split()
            label = label_line.strip().split()
            assert len(sentence) == len(label)
            sentences.append(sentence)
            labels.append(label)
        return sentences, labels

def read_types(file_path='./data/conll/types.txt'):
    """ for conll dataset
            LOC
            ORG
            PER
            MISC

    """
    type_file = open(file=file_path)
    
    # store types in list 
    types = [line.strip() for line in type_file.readlines()]
    entities_type = [] # store each 
    for entity_type in types:
        entities_type.append(entity_type.split('_')[-1])
    return entities_type

def get_data(dataset_name, data_dir):
        sentences, labels = read_file(
            data_dir, dataset_name)
        sent_len = [len(sent) for sent in sentences]
        print(f"****** {dataset_name} set stats (before tokenization): sentence length: {np.average(sent_len)} (avg) / {np.max(sent_len)} (max) ******")
        data = []
        for sentence, label in zip(sentences, labels):
            text = ' '.join(sentence)
            label = label
            data.append((text, label))
        return data

def get_label_map(file_types='./data/conll/types.txt'):
    entity_types = read_types(file_path=file_types)
    label_map = {'O': 0}
    num_labels = 1
    for entity_type in entity_types:
        label_map['B-' + entity_type] = num_labels
        label_map['I-' + entity_type] = num_labels
        num_labels += 1
    label_map['UNK'] = -100
    inv_label_map = {k: v for v, k in label_map.items()}
    return label_map, inv_label_map
     
def get_tensor(dataset_name="conll", data_dir='data/conll', max_seq_length=None, drop_o_ratio=0, tokenizer=None, file_types ='./data/conll/types.txt'):
    data_file = os.path.join(
        data_dir, f"{dataset_name}.pt")
    label_map, inv_label_map = get_label_map(file_types=file_types)
    if os.path.exists(data_file):
        print(f"Loading data from {data_file}")
        tensor_data = torch.load(data_file)
    else:
        all_data = get_data(data_dir=data_dir, dataset_name=dataset_name)
        raw_labels = [data[1] for data in all_data]
        #time to encode data 
        all_input_ids = []
        all_att_mask = []
        all_labels = []
        all_valid_pos = []
        #convert text to tensor:
        for text, labels in tqdm(all_data):
            #1. Tokenize text in data by tokenization transfromer and create mask for BERT fineturning
            encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_seq_length,
                                                        padding='max_length', return_attention_mask=True,
                                                        truncation=True, return_tensors='pt')
            
            input_ids = encoded_dict['input_ids']
            att_mask = encoded_dict['attention_mask']
            all_att_mask.append(att_mask)
            all_input_ids.append(input_ids)
            #padding label with short sentences
            labels_ids = -100*torch.ones(max_seq_length, dtype=torch.long)
            valid_pos = torch.zeros(max_seq_length, dtype=torch.long)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            # Mappling label with data 
            j = 0
            for i, token in enumerate(tokens[1:], start=1):  # skip [CLS]
                if token == tokenizer.sep_token:
                    break
                if i == 1 or token.startswith('Ġ'):
                    label = labels[j]
                    labels_ids[i] = label_map[label]
                    valid_pos[i] = 1
                    j += 1
            assert j == len(labels) or i == max_seq_length - 1
            all_labels.append(labels_ids.unsqueeze(0))
            all_valid_pos.append(valid_pos.unsqueeze(0))
        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_attention_mask = torch.cat(all_att_mask, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_valid_pos = torch.cat(all_valid_pos, dim=0)
        all_idx = torch.arange(all_input_ids.size(0))
        tensor_data = {"all_idx": all_idx, "all_input_ids": all_input_ids, "all_attention_mask": all_attention_mask,
                           "all_labels": all_labels, "all_valid_pos": all_valid_pos, "raw_labels": raw_labels}
        print(f"Saving data to {data_file}")
        torch.save(tensor_data, data_file)
    return drop_o(tensor_data=tensor_data, drop_o_ratio=drop_o_ratio)

def drop_o(tensor_data, drop_o_ratio=0):
        if drop_o_ratio == 0:
            return tensor_data
        labels = tensor_data["all_labels"]
        rand_num = torch.rand(labels.size())
        drop_pos = (labels == 0) & (rand_num < drop_o_ratio)
        labels[drop_pos] = -100
        tensor_data["all_labels"] = labels
        return tensor_data



tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case = False)
sentences, labels = read_file(file_dir='data/conll', dataset_name='train')
entities_types = read_types()
data = get_data(dataset_name='conll',data_dir='data/conll')
tensor_data = get_tensor(dataset_name='conll',data_dir='data/conll',max_seq_length=100, drop_o_ratio=0.5, tokenizer=tokenizer)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

all_idx = tensor_data["all_idx"]
all_input_ids = tensor_data["all_input_ids"]
all_attention_mask = tensor_data["all_attention_mask"]
all_labels = tensor_data["all_labels"]
all_valid_pos = tensor_data["all_valid_pos"]
tensor_data = tensor_data
gce_bin_weight = torch.ones_like(
    all_input_ids).to(device).float()
gce_type_weight = torch.ones_like(
    all_input_ids).to(device).float()

train_batch_size = 32

train_data = TensorDataset(
all_idx, all_input_ids, all_attention_mask, all_valid_pos, all_labels)

print("***** Training stats *****")
print(f"Num data = {all_input_ids.size(0)}")
print(f"Batch size = {train_batch_size}")
print("All keys of dataset", tensor_data.keys)
print("senstence data: ", sentences[0])
print("label of sentences: ", labels[0])
print("types of entities", entities_types)
print("sample of dataset: ,", data[:2])


