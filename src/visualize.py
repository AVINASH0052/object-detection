import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import SSD
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random


def visualize_predictions(image_path, model, device, confidence_threshold=0.5):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        loc_preds, cls_preds = model(input_tensor)
    
    # Convert predictions to numpy
    loc_preds = loc_preds[0].cpu().numpy()
    cls_preds = torch.softmax(cls_preds[0], dim=1).cpu().numpy()
    
    # Get class names
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    
    # Create figure and plot image
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Plot predictions
    for i in range(len(loc_preds)):
        # Get class scores
        scores = cls_preds[i]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > confidence_threshold:
            # Get box coordinates
            box = loc_preds[i]
            x1, y1, x2, y2 = box
            
            # Scale coordinates to image size
            width, height = image.size
            x1 = x1 * width
            y1 = y1 * height
            x2 = x2 * width
            y2 = y2 * height
            
            # Create rectangle patch
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Add class label and confidence
            label = f'{classes[class_id]}: {confidence:.2f}'
            ax.text(x1, y1-5, label, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    return fig


def test_model_on_random_images(model, device, num_images=4):
    # Get list of all images
    image_dir = 'data/VOC2012/JPEGImages'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # Randomly select images
    selected_images = random.sample(image_files, num_images)
    
    # Create a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, image_file in enumerate(selected_images):
        image_path = os.path.join(image_dir, image_file)
        pred_fig = visualize_predictions(image_path, model, device)
        
        # Copy the content to the subplot
        axes[idx].imshow(Image.open(image_path))
        axes[idx].axis('off')
        axes[idx].set_title(f'Image: {image_file}')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSD(num_classes=20).to(device)
    
    # Load the final checkpoint
    checkpoint_path = 'outputs/weights/ssd_model_epoch_4.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from {checkpoint_path}')
    else:
        print(f'Checkpoint not found at {checkpoint_path}')
        exit(1)
    
    # Test model on random images
    print('Testing model on random images...')
    test_model_on_random_images(model, device) 