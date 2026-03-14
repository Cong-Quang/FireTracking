import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    """
    Creates and returns train, validation, and test dataloaders.
    Applies data augmentation for training, and resizing/normalization for all.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # 1. Resize images to standard size (224x224)
    # 2. Normalize pixel values (ImageNet standards)
    # 3. Apply Data Augmentation (flip, rotation) for training
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), # Handles zoom
        transforms.RandomHorizontalFlip(), # Handles flip
        transforms.RandomRotation(15), # Handles rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names

if __name__ == "__main__":
    # Test script to verify dataloaders
    train_loader, val_loader, test_loader, classes = get_data_loaders("dataset")
    print(f"Classes: {classes}")
    
    for images, labels in train_loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        break
