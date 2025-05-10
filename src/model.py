import torch
import torch.nn as nn
import torchvision.models as models


class SSD(nn.Module):
    def __init__(self, num_classes=20):
        super(SSD, self).__init__()
        # Load pre-trained ResNet50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc layers
        
        # Detection head for box coordinates
        self.loc_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=3, padding=1)  # 4 for box coordinates
        )
        
        # Detection head for class scores
        self.cls_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=3, padding=1)  # num_classes for class scores
        )

    def forward(self, x):
        features = self.backbone(x)
        loc = self.loc_head(features)
        cls = self.cls_head(features)
        
        # Reshape outputs to match target format
        batch_size = x.size(0)
        loc = loc.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        cls = cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, cls.size(1))
        
        return loc, cls 