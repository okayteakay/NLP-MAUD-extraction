import json
import torch
from tqdm.auto import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset


def tokenize(tokenizer, data):
    # Tokenize and prepare data
    input_ids = []
    attention_masks = []
    labels = []
    
    progress_bar = tqdm(range(len(data)))
    
    for question, segment, label in data:
        encoded_dict = tokenizer.encode_plus(
        text=question, 
        text_pair=segment,
        add_special_tokens=True, 
        max_length=512,
        truncation=True, 
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(label)
        
        progress_bar.update(1)
        
    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    return dataset


def preprocessing(tokenizer, dataset, deal_points, train=True):
    segment_size = 512
    
    with open(dataset, 'r') as file:
        json_data = json.load(file)
        
    data = json_data['data']
    processed_data = []
    
    progress_bar = tqdm(range(len(data)))
    
    for example in data:
        for item in example['paragraphs']:
            context = item['context'].split()
            
            for i in range(0, len(context), segment_size):
                segment_end = min(i + segment_size, len(context))
                segment = context[i : segment_end]
                tokenized_segment = tokenizer.encode(segment, add_special_tokens=False)
               
                for qa in item['qas']:
                    if contains_substring(deal_points, qa['id']):
                        decoded_segment = tokenizer.decode(tokenized_segment, skip_special_tokens=True)

                        question = qa['question']
                        
                        for answer in qa['answers']:
                            if segment_contains_answer(tokenizer, context, i, i + segment_size, answer['answer_start'], answer['text']):
                                # Label segment as relevant to the question
                                processed_data.append((question, decoded_segment, 1))
                            else:
                                # Label segment as irrelevant to the question
                                processed_data.append((question, decoded_segment, 0))
        progress_bar.update(1)
        
    output_fp = './out/dataset_preprocessed/maud_class_train.json' if train else './out/dataset_preprocessed/maud_class_test.json'
    
    # Convert dataset to a list of dictionaries
    data_to_write = [{'question': item[0], 'segment': item[1], 'label': item[2]} for item in processed_data]
    
    # Write to JSON file
    with open(output_fp, 'w') as file:
        json.dump(data_to_write, file, indent=4)
                        
    return processed_data


# Checks if a string contains any substring form a list of substrings
def contains_substring(substrings, string):
    for sub in substrings:
        if sub in string:
            return True
        
    return False


# Checks if a segment contains an answer
def segment_contains_answer(tokenizer, context, segment_start, segment_end, answer_start, answer):
    answer_end = answer_start + len(answer)
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
    