import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VOCDataset, collate_fn
from model import SSD
import os
import torch.nn.functional as F
from tqdm import tqdm


def compute_loss(loc_preds, loc_targets, cls_preds, cls_targets):
    # Reshape predictions
    batch_size = loc_preds.size(0)
    num_priors = loc_preds.size(1)
    
    # Create target tensors
    loc_target = torch.zeros_like(loc_preds)
    cls_target = torch.zeros((batch_size, num_priors), dtype=torch.long, device=loc_preds.device)
    
    # Match predictions with ground truth
    for i in range(batch_size):
        num_objects = len(loc_targets[i])
        if num_objects > 0:
            # Get ground truth boxes and predictions for this image
            gt_boxes = loc_targets[i]  # [num_objects, 4]
            pred_boxes = loc_preds[i]   # [num_priors, 4]
            
            # Calculate IoU between all predictions and ground truth
            ious = []
            for gt_box in gt_boxes:
                # Expand gt_box to match pred_boxes dimensions
                gt_box = gt_box.unsqueeze(0).expand_as(pred_boxes)
                iou = calculate_iou(pred_boxes, gt_box)
                ious.append(iou)
            
            # Stack IoUs and find best matches
            ious = torch.stack(ious, dim=0)  # [num_objects, num_priors]
            best_prior_idx = ious.argmax(dim=1)  # [num_objects]
            
            # Update targets
            for obj_idx, prior_idx in enumerate(best_prior_idx):
                loc_target[i, prior_idx] = gt_boxes[obj_idx]
                # Convert tensor to integer for indexing
                prior_idx = prior_idx.item()
                # Convert class label to integer
                if isinstance(cls_targets[i], torch.Tensor):
                    if cls_targets[i].dim() == 0:
                        cls_label = cls_targets[i].item()
                    else:
                        cls_label = cls_targets[i][obj_idx].item()
                else:
                    cls_label = cls_targets[i][obj_idx]
                cls_target[i, prior_idx] = cls_label
    
    # Compute losses
    pos_mask = cls_target > 0
    loc_loss = F.smooth_l1_loss(loc_preds[pos_mask], loc_target[pos_mask], reduction='sum')
    cls_loss = F.cross_entropy(cls_preds.view(-1, cls_preds.size(-1)), 
                             cls_target.view(-1), 
                             reduction='sum')
    
    # Normalize by number of positive matches
    num_pos = pos_mask.sum().float()
    loc_loss = loc_loss / num_pos if num_pos > 0 else loc_loss
    cls_loss = cls_loss / num_pos if num_pos > 0 else cls_loss
    
    return loc_loss, cls_loss


def calculate_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    boxes1: [N, 4] where N is the number of boxes
    boxes2: [N, 4] where N is the number of boxes
    """
    # Calculate intersection coordinates
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    # Calculate intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union area
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    return iou


def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directories
    os.makedirs('outputs/weights', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)

    # Dataset and DataLoader
    root_dir = 'data/VOC2012'
    train_split_file = os.path.join(root_dir, 'ImageSets/Main/train.txt')
    print(f'Loading dataset from {root_dir}...')
    train_dataset = VOCDataset(root_dir=root_dir, split_file=train_split_file)
    print(f'Dataset size: {len(train_dataset)} images')
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
    print(f'Number of batches per epoch: {len(train_loader)}')

    # Model
    print('Initializing model...')
    model = SSD(num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Training loop
    num_epochs = 5
    print(f'Starting training for {num_epochs} epochs...')
    
    for epoch in range(num_epochs):
        model.train()
        running_loc_loss = 0.0
        running_cls_loss = 0.0
        
        # Create progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, boxes, labels) in enumerate(pbar):
            images = images.to(device)
            
            # Process each image in the batch
            batch_loc_loss = 0.0
            batch_cls_loss = 0.0
            num_images = len(images)
            
            for i in range(num_images):
                # Get predictions for single image
                loc_preds, cls_preds = model(images[i:i+1])
                
                # Get ground truth for single image
                img_boxes = boxes[i].to(device)
                img_labels = labels[i].to(device)
                
                # Compute losses
                loc_loss, cls_loss = compute_loss(loc_preds, img_boxes, cls_preds, img_labels)
                batch_loc_loss += loc_loss
                batch_cls_loss += cls_loss
            
            # Average losses over batch
            batch_loc_loss /= num_images
            batch_cls_loss /= num_images
            total_loss = batch_loc_loss + batch_cls_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loc_loss += batch_loc_loss.item()
            running_cls_loss += batch_cls_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loc_loss': f'{batch_loc_loss.item():.4f}',
                'cls_loss': f'{batch_cls_loss.item():.4f}',
                'total_loss': f'{total_loss.item():.4f}'
            })
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch statistics
        avg_loc_loss = running_loc_loss / len(train_loader)
        avg_cls_loss = running_cls_loss / len(train_loader)
        print(f'\nEpoch {epoch+1}/{num_epochs} completed:')
        print(f'Average Localization Loss: {avg_loc_loss:.4f}')
        print(f'Average Classification Loss: {avg_cls_loss:.4f}')
        print(f'Average Total Loss: {avg_loc_loss + avg_cls_loss:.4f}')
        
        # Save model checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f'outputs/weights/ssd_model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')


if __name__ == '__main__':
    train() 