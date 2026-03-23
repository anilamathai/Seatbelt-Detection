import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from utils import load_config, save_checkpoint

def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = config["training"]["image_size"]

    #Data augmentation:::
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    train_dir = config["paths"]["data_dir"]

    #Load dataset:::
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    #Model:::
    model = models.mobilenet_v2(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 3)  # 3 classes: Seatbelt, No-Seatbelt, VLQ
    model = model.to(device)

    #Loss with label smoothing:::
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    #Optimizer:::
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    best_loss = float("inf")

    #Training loop:::
    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {epoch_loss:.4f}")

        #Save best model:::
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, f"{config['paths']['checkpoint_dir']}/best_model.pth")

if __name__ == "__main__":
    main()
