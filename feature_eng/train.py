import os
import torch
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torchmetrics import PrecisionRecallCurve
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from utils import preprocessing, tokenize, oversample_minority, load_json


def train(model, device, train_dataloader, val_dataloader, save_dir="./out"):
    file = open(save_dir + "/results.txt", 'w')
    divider = "-" * 12
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-5, no_deprecation_warning=True)
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
        avg_train_loss = total_loss / len(train_dataloader)
        epoch_train_loss = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss}"
        print(epoch_train_loss)
        file.write(epoch_train_loss)

        val_accuracy = evaluate(model, val_dataloader, device)
        val_acc_text = f"Validation Accuracy: {val_accuracy}"
        print(val_acc_text)
        file.write(val_acc_text)
        
        file.write(divider)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(save_dir + '/model/')
            
    best_val_text = f"Training complete. Best Validation Accuracy: {best_val_accuracy}"
    print(best_val_text)
    file.write(best_val_text)
    
    file.close()
    
    
def evaluate(model, val_dataloader, device):
    model.to(device)
    
    model.eval()
    
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = probabilities[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    aupr_score = auc(recall, precision)
    
    return aupr_score


if __name__ == "__main__":
    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    deal_points = [
        "General Antitrust Efforts Standard", 
        "Intervening Event Definition", 
        "Knowledge Definition", 
        "Negative interim operating covenant", 
        "Specific Performance",
        "Type of Consideration"
        ]
    train_dataset = "maud_data/maud_squad_split_answers/maud_squad_train.json"
    val_dataset = "maud_data/maud_squad_split_answers/maud_squad_test.json"
    
    # Load pre-trained LegalBERT model and tokenizer
    legalbert = 'nlpaueb/legal-bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(legalbert)
    tokenizer = BertTokenizer.from_pretrained(legalbert, do_lower_case=True)
    
    # Preprocess the training data
    # print("Preprocessing training data...")
    # preprocessed_train_data = preprocessing(tokenizer, train_dataset, deal_points)
    print("Loading training data...")
    train_class_data = "./out/dataset_preprocessed/maud_class_test.json"
    preprocessed_train_data = load_json(train_class_data)
    
    # Oversample minority class
    print("Oversampling minority class...")
    oversampled_train_data = oversample_minority(preprocessed_train_data)
    
    print("Tokenizing training data...")
    # Tokenize the training data
    tokenized_train_data = tokenize(tokenizer, oversampled_train_data)
    
    # print("Preprocessing test data")
    # # Preprocess the validation data
    # preprocessed_val_data = preprocessing(tokenizer, val_dataset, deal_points, train=False)
    print("Loading test data...")
    test_class_data = "./out/dataset_preprocessed/maud_class_test.json"
    preprocessed_val_data = load_json(test_class_data)
    
    print("Tokenize test data...")
    # Tokenize the validation data
    tokenized_val_data = tokenize(tokenizer, preprocessed_val_data, train=False)
    
    # Create training set dataloader
    dataloader_train = DataLoader(tokenized_train_data, shuffle=True, batch_size=8)
    
    # Create validation set dataloader
    dataloader_val = DataLoader(tokenized_val_data, batch_size=8)
    
    # Train model
    print("Training model...")
    train(model, device, dataloader_train, dataloader_val)
    
    # Save tokenizer
    tokenizer.save_pretrained('./out/tokenizer/')
