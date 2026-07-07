"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
import torch.nn as nn

#activation_str = "ReLU"  # moved to config file


class VGGBlock(nn.Module):
    """Modular VGG block with configurable number of conv layers and channels.

    C configuration from Simonyan & Zisserman's VGG paper.
    """
    def __init__(self, in_channels, out_channels, num_convs, padding=1):
        super().__init__()
        layers = []
        current_in_channels = in_channels
        for i in range(num_convs):
            is_config_c_tail = (num_convs == 3 and i == 2)
            kernel_size = 1 if is_config_c_tail else 3
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_in_channels = out_channels     

            
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """ResBlock with 3x3 convolutions (He et al., 2016)."""
    def __init__(self, in_channels, out_channels, activation, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If spatial size shrinks (stride > 1) or channels change, adjust the shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  
        out = self.activation(out)
        return out


class AlexNet(nn.Module):
    """AlexNet (Krizhevsky et al., 2012) adapted for smaller inputs."""
    def __init__(self, in_channels, num_classes, **kwargs):   #added in_channels and ouput chnannels 
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3),   #added in_channels 
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                       #added Adativeaverage pooling layer to reduce  dimensions to 1x1.
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(192, 1024),                                         #2048 -> 192 to match the output channels of the last conv layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),                                     #added num_classes to the final layer to match the number of output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)                                        # Apply adaptive average pooling
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGG16(nn.Module):
    """VGG16 in C configuration of Simonyan & Zisserman, (2014) adapted for smaller inputs."""
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)

        self.features = nn.Sequential(
            VGGBlock(in_channels, 64, num_convs=2),
            VGGBlock(64, 128, num_convs=2),
            VGGBlock(128, 256, num_convs=3),
            VGGBlock(256, 512, num_convs=3),
            VGGBlock(512, 512, num_convs=3)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))        # added adaptive average pooling layer to reduce the dimensions to 1x1.
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),                          # 2048 -> 512 to match the output channels of the last conv layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)                                        # Apply adaptive average pooling
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet18(nn.Module):
    """ResNet18 (He et al., 2016) adapted for smaller inputs.
    
    activation - flexible activation function to allow experimentation (e.g., ReLU, LeakyReLU, etc.)
    """
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        activation = getattr(nn, kwargs.get("activation_str") or "ReLU") # read form config file

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = activation(inplace=True)
        print("Using activation function:", self.activation)
        
        self.stage1 = nn.Sequential(
            ResBlock(64, 64, activation(inplace=True), stride=1),
            ResBlock(64, 64, activation(inplace=True), stride=1)
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, activation(inplace=True), stride=2),          
            ResBlock(128, 128, activation(inplace=True), stride=1)
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, activation(inplace=True), stride=2),
            ResBlock(256, 256, activation(inplace=True), stride=1)
        )
        self.stage4 = nn.Sequential(
            ResBlock(256, 512, activation(inplace=True), stride=2),
            ResBlock(512, 512, activation(inplace=True), stride=1)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)   # missing return  


# reduced green initiative model
class GreenResNet(nn.Module):
    """Green model = TWO efficiency techniques from the lecture notes:
       1) reduced channels (32 base instead of 64)  - Lecture Notes 8.2.6
       2) bottleneck blocks (1x1 squeeze -> 3x3 -> 1x1 restore) - Lecture Notes 8.4.4
       The bottleneck is built inline via _make_stage; no separate block class.
    """
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        act = getattr(nn, kwargs.get("activation_str") or "ReLU")  # read from config
        self.act = act
        self.activation = act(inplace=True)
 
        # stem: reduced to 32 channels (ResNet18 uses 64)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
 
        # reduced-channel stages, each = 2 bottleneck sub-blocks
        self.stage1 = self._make_stage(32, 32, stride=1)
        self.stage2 = self._make_stage(32, 64, stride=2)
        self.stage3 = self._make_stage(64, 128, stride=2)
        self.stage4 = self._make_stage(128, 256, stride=2)
 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)
 
    def _bottleneck(self, in_c, out_c, stride, reduction=4):
        """1x1 squeeze -> 3x3 process -> 1x1 restore, with residual shortcut."""
        mid = max(out_c // reduction, 8)
        body = nn.Sequential(
            nn.Conv2d(in_c, mid, 1, bias=False), nn.BatchNorm2d(mid), self.act(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(mid), self.act(inplace=True),
            nn.Conv2d(mid, out_c, 1, bias=False), nn.BatchNorm2d(out_c),
        )
        shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c)
            )
        return nn.ModuleDict({"body": body, "shortcut": shortcut})
 
    def _make_stage(self, in_c, out_c, stride):
        """A stage = two bottleneck sub-blocks; the first handles downsampling."""
        return nn.ModuleList([
            self._bottleneck(in_c, out_c, stride=stride),
            self._bottleneck(out_c, out_c, stride=1),
        ])
 
    def _run_stage(self, stage, x):
        for blk in stage:
            x = self.activation(blk["body"](x) + blk["shortcut"](x))
        return x
 
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self._run_stage(self.stage1, out)
        out = self._run_stage(self.stage2, out)
        out = self._run_stage(self.stage3, out)
        out = self._run_stage(self.stage4, out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)
