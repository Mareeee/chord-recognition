import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.model import CNNChordNet

def evaluate_model(model, loader, device):
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad(): 
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    model.train()
    return accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        val_accuracy = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1} finished. Loss: {epoch_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

def test_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing on Test Set"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    return test_accuracy

def run_training_evaluation(dataset_instance, num_classes, device, epochs, model_path):
    
    total_size = len(dataset_instance)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    sizes = [train_size, val_size, test_size]
    if sum(sizes) != total_size:
        test_size += total_size - sum(sizes)
        sizes = [train_size, val_size, test_size]

    train_dataset, val_dataset, test_dataset = random_split(
        dataset_instance, sizes
    )

    print(f"Data Split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CNNChordNet(num_classes=num_classes).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting training for {epochs} epochs...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs)

    print("Training finished.")

    print("\n--- FINAL EVALUATION ON TEST SET ---")
    final_test_accuracy = test_model(model, test_loader, device)

    print(f"Final Test Accuracy (Approximation of CSR): {final_test_accuracy:.2f}%")
    
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved successfully to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")