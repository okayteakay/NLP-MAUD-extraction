import json
import torch
import transformers
import argparse
from transformers import BertTokenizer, BertForSequenceClassification


def predict(model, device, input, tokenizer, passages):
    model.to(device)
    model.eval()
    
    all_logits = []
    
    with torch.no_grad():
        for qp in input:
            output = model(**qp).logits
            all_logits.append(output)
            
    probabilities = [ torch.nn.functional.softmax(logit, dim=1)[:, 1] for logit in all_logits ]
    
    prob_passage_pairs = [ (prob[1].item(), tokenizer.decode(passage, skip_special_tokens=True)) for prob, passage in zip(probabilities, passages) ]
    
    sorted_pairs = sorted(prob_passage_pairs, key=lambda x: x[0], reverse=True)
    
    return sorted_pairs


def tokenize_input(tokenizer, query, document):
    document_tokens = tokenizer.encode(document, add_special_tokens=False)
    max_length = 512
    max_chunk_length = max_length - len(tokenizer.encode(query, add_special_tokens=True)) - 1  # for [SEP]
    
    passages = [ document_tokens[i:i + max_chunk_length] for i in range(0, len(document_tokens), max_chunk_length) ]

    encoded_inputs = [ 
                      tokenizer.encode_plus(
                          text=query,
                          text_pair=tokenizer.decode(passage),
                          add_special_tokens=True,
                          truncation=True,
                          max_length=max_length,
                          padding='max_length',
                          return_attention_mask=True,
                          return_tensors='pt'
                      ) for passage in passages ]
    
    return encoded_inputs, passages


def get_inputs(file):
    with open(file, 'r') as json_file:
        json_data = json.load(json_file)
        
    return json_data['question'], json_data['context']
        
        


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input",
        default=None,
        type=str,
        required=True,
        help="Input JSON file"
    )
    
    parser.add_argument(
        "--output",
        default="ranked_results.txt",
        type=str,
        required=False,
        help="Output file for ranked results."
    )
    
    parser.add_argument(
        "--model_dir",
        default="../out/model",
        type=str,
        required=False,
        help="File directory containing model."
    )
    
    parser.add_argument(
        "--tokenizer_dir",
        default="../out/tokenizer",
        type=str,
        required=False,
        help="File directory containing tokenizer."
    )
    
    parser.add_argument(
        "--k",
        default=10,
        type=int,
        required=False,
        help="Return top k results."
    )
    
    # parser.add_argument(
    #     "--is_squad",
    #     action="store_true",
    #     help="If the input JSON is of SQuAD format."
    # )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_dir)
    
    query, document = get_inputs(args.input)
    
    inputs, passages = tokenize_input(tokenizer, query, document)
    
    predictions = predict(model, device, inputs, tokenizer, passages)
    
    file = open(args.output, 'w')
    
    query_text = f"Question: {query}\n"
    file.write(query_text)
    print(query_text)
    
    i = 1
    
    for prob, passage in predictions[:args.k]:
        index_text = f"{i})\n"
        prob_text = f"Score: {prob}\n"
        passage_text = f"Passage: {passage}\n"
        
        file.write(index_text)
        file.write(prob_text)
        file.write(passage_text)
        
        print(index_text)
        print(prob_text)
        print(passage_text)
        
        i += 1
        
    file.close()
    
    
    
    