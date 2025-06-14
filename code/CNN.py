from prepros import clean_text, preprocess_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch.optim as optim
from nltk import word_tokenize
#import nltk
#nltk.download('punkt')

def build_vocab(texts, max_vocab_size=10000):
    word_counts = Counter()
    for tokens in texts:
        word_counts.update(tokens)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab.update({word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common(max_vocab_size - 2))})
    return vocab

def index(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

class sms_dataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=200):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.vocab = vocab

    def __len__ (self):
        return len(self.texts)

    def __getitem__ (self, idx):
        tokens = self.texts[idx]
        label = self.labels[idx]
        indices = index(tokens, self.vocab)

        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=128, filter_sizes=[2, 3, 4], num_classes=2, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  
        x = x.permute(0, 2, 1)  
        conv_outputs = [nn.functional.relu(conv(x)) for conv in self.convs] 
        pooled = [nn.functional.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in conv_outputs]
        cat = torch.cat(pooled, dim=1) 
        cat = self.dropout(cat)
        logits = self.fc(cat) 
        return logits


def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(loader), correct / total


def main():
    # Preprocess data
    df = preprocess_data()
    X = df['text'].values
    y = df['label'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Build vocab
    vocab = build_vocab(X_train)
    
    # Create datasets
    max_length = 100
    train_dataset = sms_dataset(X_train, y_train, vocab, max_length)
    val_dataset = sms_dataset(X_val, y_val, vocab, max_length)
    test_dataset = sms_dataset(X_test, y_test, vocab, max_length)

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextCNN(vocab_size=len(vocab), embed_dim=100, num_filters=128, filter_sizes=[2, 3, 4], num_classes=2)
    model.to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device, epochs=10)
    
    # Evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    spam_text = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."
    real_text = "Congratulations! You've won my heart. Come to my place to claim your prize."
    spam_prediction = predict(spam_text, model, vocab, max_length, device)
    real_prediction = predict(real_text, model, vocab, max_length, device)
    print(f'Prediction for real text: "{real_text}" is: {real_prediction}')
    print(f'Prediction for test text: "{spam_text}" is: {spam_prediction}')

def predict(text, model, vocab, max_length, device):
    model.eval()
    text = clean_text(text)  
    tokens = word_tokenize(text)  
    indices = index(tokens, vocab)
    if len(indices) < max_length:
        indices = indices + [vocab['<PAD>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    text_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(text_tensor)
        _, predicted = torch.max(output, 1)
    return 'spam' if predicted.item() == 1 else 'not spam'        



if __name__ == "__main__":
    main()













