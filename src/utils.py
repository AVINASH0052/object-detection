import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_voc_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        label = obj.find('name').text
        labels.append(label)
    return boxes, labels


def get_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def visualize_detection(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline='red', width=2)
        draw.text((box[0], box[1]), label, fill='red')
    return image


def visualize_predictions(image_path, model, device, confidence_threshold=0.5, iou_threshold=0.5):
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
    
    loc_preds = loc_preds[0].cpu()
    cls_preds = torch.softmax(cls_preds[0], dim=1).cpu()
    
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    
    boxes = []
    scores = []
    labels = []
    width, height = image.size

    for i in range(len(loc_preds)):
        score = torch.max(cls_preds[i]).item()
        class_id = torch.argmax(cls_preds[i]).item()
        if score > confidence_threshold:
            x1, y1, x2, y2 = loc_preds[i]
            x1 = x1.item() * width
            y1 = y1.item() * height
            x2 = x2.item() * width
            y2 = y2.item() * height
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(class_id)
    
    if boxes:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores)
        labels = torch.tensor(labels)
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
    else:
        boxes = torch.empty((0, 4))
        scores = torch.empty((0,))
        labels = torch.empty((0,), dtype=torch.int64)
    
    # Plot
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label = f'{classes[labels[i]]}: {scores[i]:.2f}'
        ax.text(x1, y1-5, label, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    plt.axis('off')
    plt.show() 