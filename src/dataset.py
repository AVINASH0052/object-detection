import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def collate_fn(batch):
    images = []
    boxes = []
    labels = []
    
    for image, box, label in batch:
        images.append(image)
        boxes.append(box)
        labels.append(label)
    
    images = torch.stack(images, 0)
    return images, boxes, labels


class VOCDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        
        # Get available image IDs from the directory
        self.available_images = set()
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.jpg'):
                self.available_images.add(filename[:-4])  # Remove .jpg extension
        
        # Load image IDs from split file
        self.image_ids = []
        with open(split_file, 'r') as f:
            for line in f:
                img_id = line.strip()
                # Try both 2008 and 2012 versions of the image ID
                if img_id in self.available_images:
                    self.image_ids.append(img_id)
                else:
                    # Try converting 2008 to 2012 format
                    year = img_id.split('_')[0]
                    if year == '2008':
                        new_id = '2012_' + '_'.join(img_id.split('_')[1:])
                        if new_id in self.available_images:
                            self.image_ids.append(new_id)
        
        print(f'Found {len(self.image_ids)} valid images in the dataset')
        
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_id + '.jpg')
        annotation_path = os.path.join(self.annotation_dir, img_id + '.xml')
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Parse annotations
        boxes, labels = self._parse_annotation(annotation_path)
        
        # Convert boxes to tensor if not already
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
            
        return image, boxes, labels

    def _parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        
        # Get image size for normalization
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / width
            ymin = float(bbox.find('ymin').text) / height
            xmax = float(bbox.find('xmax').text) / width
            ymax = float(bbox.find('ymax').text) / height
            boxes.append([xmin, ymin, xmax, ymax])
            label = obj.find('name').text
            labels.append(self.class_to_idx[label])
            
        return boxes, labels 