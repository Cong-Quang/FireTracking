import torch
import torch.nn as nn
from dataset import get_data_loaders
from models import get_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def evaluate_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model '{model_name}' on device: {device}")
    
    # Needs test dataloader setup from earlier step
    _, _, test_loader, class_names = get_data_loaders('dataset', batch_size=32)
    num_classes = len(class_names)
    
    # Load model
    model = get_model(model_name, num_classes=num_classes)
    weights_path = f"checkpoints/{model_name}_best.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Could not find weights at {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_name}')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    print(f"Saved confusion matrix to results/{model_name}_confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='custom_cnn', 
                        choices=['custom_cnn', 'resnet50', 'mobilenet_v2', 'vgg16', 'efficientnet'])
    args = parser.parse_args()
    
    evaluate_model(args.model)
