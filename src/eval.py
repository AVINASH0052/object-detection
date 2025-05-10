import torch
from torch.utils.data import DataLoader
from dataset import VOCDataset
from model import SSD
import os


def evaluate():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Dataset and DataLoader
    root_dir = 'data/VOC2012'
    val_split_file = os.path.join(root_dir, 'ImageSets/Main/val.txt')
    val_dataset = VOCDataset(root_dir=root_dir, split_file=val_split_file)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Load model
    model = SSD(num_classes=20).to(device)
    model.load_state_dict(torch.load('outputs/weights/ssd_model.pth'))
    model.eval()

    # Evaluation loop
    total_loss = 0.0
    with torch.no_grad():
        for images, boxes, labels in val_loader:
            images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)
            outputs = model(images)
            loss = torch.nn.functional.mse_loss(outputs, boxes)  # Example loss calculation
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}')


if __name__ == '__main__':
    evaluate() 