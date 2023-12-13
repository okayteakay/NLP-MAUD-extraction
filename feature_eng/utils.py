import json
import torch
from tqdm.auto import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from imblearn.over_sampling import RandomOverSampler
import numpy as np


def tokenize(tokenizer, data, train=True):
    # Tokenize and prepare data
    input_ids = []
    attention_masks = []
    labels = []
    
    if train:
        progress_bar = tqdm(range(len(data)))
        
        for question, text, label in data:
            encoded_dict = tokenizer.encode_plus(
            text=question, 
            text_pair=text,
            add_special_tokens=True,
            max_length=512, 
            truncation=True, 
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
            
            input_ids.append(encoded_dict['input_ids'].squeeze(0))
            attention_masks.append(encoded_dict['attention_mask'].squeeze(0))
            labels.append(label)
                
            progress_bar.update(1)
            
    else:
        progress_bar = tqdm(range(len(data)))
        
        for contract, qps in data.items():
            for qp in qps:
                question = qp['question']
                passages = qp['passages']
                for tl in passages:
                    text = tl['text']
                    label = tl['label']
                    
                    encoded_dict = tokenizer.encode_plus(
                    text=question, 
                    text_pair=text,
                    add_special_tokens=True, 
                    truncation=True, 
                    max_length=512,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                    
                    input_ids.append(encoded_dict['input_ids'].squeeze(0))
                    attention_masks.append(encoded_dict['attention_mask'].squeeze(0))
                    labels.append(label)
                
            progress_bar.update(1)
    
    labels = np.array(labels)
    labels = labels.astype(int)
        
    # Convert lists to tensors
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    return dataset



def oversample_minority(data):
    processed_data = []
    
    for contract, qps in data.items():
        for qp in qps:
            question = qp['question']
            passages = qp['passages']
            for tl in passages:
                text = tl['text']
                label = tl['label']
                processed_data.append([question, text, label])
                
    structured_array = np.array(processed_data)
    X = structured_array[:,:2]
    y = structured_array[:,2]
                
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    resampled_array = np.column_stack((X_resampled, y_resampled))
    
    return resampled_array


def preprocessing(tokenizer, dataset, deal_points, train=True, test=False):
    segment_size = 512
    
    with open(dataset, 'r') as file:
        json_data = json.load(file)
        
    data = json_data['data']
    processed_data = {}
    
    progress_bar = tqdm(range(len(data)))
    
    for example in data:
        title = example['title']
        for item in example['paragraphs']:
            context = item['context'].split()
            
            for qa in item['qas']:
                if contains_substring(deal_points, qa['id']):
                    question = qa['question']
                    if title not in processed_data:
                        processed_data[title] = []
                    processed_data[title].append({
                        'question': question,
                        'passages': []
                    })
                    
                    for i in range(0, len(context), segment_size):
                        segment_end = min(i + segment_size, len(context))
                        segment = context[i : segment_end]
                        tokenized_segment = tokenizer.encode(segment, add_special_tokens=False)
               
                        decoded_segment = tokenizer.decode(tokenized_segment, skip_special_tokens=True)
                        
                        answers = qa['answers']
                        if segment_contains_answer(tokenizer, context, i, i + segment_size, answers):
                            # Label segment as relevant to the question
                            processed_data[title][-1]['passages'].append({ 'text': decoded_segment, 'label': 1 })
                            
                        else:
                            # Label segment as irrelevant to the question
                            processed_data[title][-1]['passages'].append({ 'text': decoded_segment, 'label': 0 })
        progress_bar.update(1)
        if test: break
        
    output_fp = './out/dataset_preprocessed/maud_class_train.json' if train else './out/dataset_preprocessed/maud_class_test.json'
    
    # Write to JSON file
    with open(output_fp, 'w') as file:
        json.dump(processed_data, file, indent=4)
                        
    return processed_data


# Checks if a string contains any substring form a list of substrings
def contains_substring(substrings, string):
    for sub in substrings:
        if sub in string:
            return True
        
    return False


# Checks if a segment contains an answer
def segment_contains_answer(tokenizer, context, segment_start, segment_end, answers):
    for answer in answers:
        answer_start = answer['answer_start']
        answer_text = answer['text']
        answer_end = answer_start + len(answer_text)
        start_token_idx = len(tokenizer.encode(context[:answer_start])) - 1
        end_token_idx = len(tokenizer.encode(context[:answer_end])) - 1
        
        # If segment contains the entire answer
        if start_token_idx >= segment_start and end_token_idx <= segment_end:
            return True
        # If segment contains part of the answer
        elif (start_token_idx >= segment_start and start_token_idx <= segment_end) or (end_token_idx >= segment_start and end_token_idx <= segment_end):
            # Ratio of partial answer text to full segment
            ratio = len(context[max(start_token_idx, segment_start) : min(end_token_idx, segment_end)]) / (segment_end - segment_start)
            # Threshold of 0.25
            if ratio >= 0.25:
                return True
    
    return False


def load_json(json_data):
    with open(json_data, 'r') as file:
        data = json.load(file)
        
    return data
    