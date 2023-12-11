import os
import torch
from tqdm.auto import tqdm
from sklearn.metrics import auc
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torchmetrics import PrecisionRecallCurve
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from utils import preprocessing, tokenize


def train(model, device, train_dataloader, val_dataloader, save_dir="./out"):
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 10
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
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss}")

        val_accuracy = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {val_accuracy}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(save_dir + '/model/')
            
            
    print(f"Training complete. Best Validation Accuracy: {best_val_accuracy}")
    
    
def evaluate(model, val_dataloader, device):
    model.to(device)
    
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs).logits
            all_preds.extend(outputs.cpu())
            all_labels.extend(labels.cpu())

    precision, recall, _ = PrecisionRecallCurve()(torch.tensor(all_preds), torch.tensor(all_labels))
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
    print("Preprocessing training data...")
    preprocessed_train_data = preprocessing(tokenizer, train_dataset, deal_points)
    
    print("Tokenizing training data")
    # Tokenize the training data
    tokenized_train_data = tokenize(tokenizer, preprocessed_train_data)
    
    print("Preprocessing test data")
    # Preprocess the validation data
    preprocessed_val_data = preprocessing(tokenizer, val_dataset, deal_points, train=False)
    
    print("Tokenize test data")
    # Tokenize the validation data
    tokenized_val_data = tokenize(tokenizer, preprocessed_val_data)
    
    # Create training set dataloader
    dataloader_train = DataLoader(tokenized_train_data, shuffle=True, batch_size=8)
    
    # Create validation set dataloader
    dataloader_val = DataLoader(tokenized_val_data, shuffle=True, batch_size=8)
    
    # Train model
    train(model, device, dataloader_train, dataloader_val)
    
    # Save tokenizer
    tokenizer.save_pretrained('./out/tokenizer/')