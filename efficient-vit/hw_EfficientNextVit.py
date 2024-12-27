# import torch
# from torch import nn
# from einops import rearrange
# from timm.models.layers import DropPath
# from functools import partial
# from efficientnet_pytorch import EfficientNet

# class PatchEmbed(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(PatchEmbed, self).__init__()
#         self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False) if stride == 2 else nn.Identity()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
#         self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.conv(x)
#         return self.norm(x)

# class MHCA(nn.Module):
#     def __init__(self, out_channels, head_dim):
#         super(MHCA, self).__init__()
#         self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels // head_dim, bias=False)
#         self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
#         self.act = nn.ReLU(inplace=True)
#         self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

#     def forward(self, x):
#         x = self.group_conv3x3(x)
#         x = self.norm(x)
#         x = self.act(x)
#         return self.projection(x)

# class Mlp(nn.Module):
#     def __init__(self, in_features, mlp_ratio=4, drop=0.):
#         super().__init__()
#         hidden_dim = int(in_features * mlp_ratio)
#         self.fc1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1)
#         self.act = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(hidden_dim, in_features, kernel_size=1)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         return self.drop(x)

# class NCB(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, path_dropout=0., drop=0., head_dim=32, mlp_ratio=4):
#         super(NCB, self).__init__()
#         self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
#         self.mhca = MHCA(out_channels, head_dim)
#         self.drop_path1 = DropPath(path_dropout) if path_dropout > 0. else nn.Identity()
#         self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
#         self.mlp = Mlp(out_channels, mlp_ratio, drop)
#         self.drop_path2 = DropPath(path_dropout) if path_dropout > 0. else nn.Identity()

#     def forward(self, x):
#         x = self.patch_embed(x)
#         x = x + self.drop_path1(self.mhca(x))
#         x = x + self.drop_path2(self.mlp(self.norm(x)))
#         return x

# class NextViTHead(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(NextViTHead, self).__init__()
#         self.norm = nn.BatchNorm2d(in_channels, eps=1e-5)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(in_channels, num_classes)

#     def forward(self, x):
#         x = self.norm(x)
#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         return self.fc(x)

# class EfficientNextViT(nn.Module):
#     def __init__(self, num_classes=1, efficientnet_type='efficientnet-b0', head_depths=[2, 2, 6, 2], head_dim=32):
#         super(EfficientNextViT, self).__init__()
#         # EfficientNet Backbone
#         self.efficient_net = EfficientNet.from_pretrained(efficientnet_type)

#         # Freeze all layers except the last few blocks
#         for param in self.efficient_net.parameters():
#             param.requires_grad = False
#         for block in self.efficient_net._blocks[-3:]:
#             for param in block.parameters():
#                 param.requires_grad = True

#         # NextViT Classification Head
#         self.head = nn.ModuleList()
#         input_channels = self.efficient_net._conv_head.out_channels
#         for depth in head_depths:
#             stage = []
#             for _ in range(depth):
#                 stage.append(NCB(input_channels, input_channels, head_dim=head_dim))
#             self.head.append(nn.Sequential(*stage))

#         self.classifier = NextViTHead(input_channels, num_classes)

#     def forward(self, x):
#         # EfficientNet feature extraction
#         x = self.efficient_net.extract_features(x)

#         # NextViT head processing
#         for stage in self.head:
#             x = stage(x)

#         # Classification
#         return self.classifier(x)

# # # Example instantiation
# # model = EfficientNextViT(num_classes=1000, efficientnet_type='efficientnet-b0', head_depths=[2, 2, 6, 2], head_dim=32)








'''Efficient_Vit(두 번째)'''
import torch
from torch import nn
from einops import rearrange
from efficientnet_pytorch import EfficientNet
from functools import partial
from timm.models.layers import DropPath

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PatchEmbed, self).__init__()
        self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False) if stride == 2 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        return self.norm(x)

class MHCA(nn.Module):
    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.group_conv3x3(x)
        x = self.norm(x)
        x = self.act(x)
        return self.projection(x)

class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4, drop=0.):
        super().__init__()
        hidden_dim = int(in_features * mlp_ratio)
        self.fc1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, 640, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

class NCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0., drop=0., head_dim=32, mlp_ratio=4):
        super(NCB, self).__init__()
        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.drop_path1 = DropPath(path_dropout) if path_dropout > 0. else nn.Identity()
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.mlp = Mlp(out_channels, mlp_ratio, drop)
        self.drop_path2 = DropPath(path_dropout) if path_dropout > 0. else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.drop_path1(self.mhca(x))
        x = x + self.drop_path2(self.mlp(self.norm(x)))
        return x

class NTB(nn.Module):
    def __init__(self, in_channels, out_channels, path_dropout=0., drop=0., head_dim=32, mlp_ratio=4):
        super(NTB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.e_mhsa = nn.MultiheadAttention(embed_dim=in_channels, num_heads=in_channels // head_dim, dropout=drop)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # K, V에 적용할 Avg Pooling
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.mhca = MHCA(out_channels, head_dim)
        self.norm3 = nn.BatchNorm2d(out_channels * 2)
        self.mlp = Mlp(out_channels * 2, mlp_ratio, drop)
        self.drop_path = DropPath(path_dropout) if path_dropout > 0. else nn.Identity()

    def forward(self, x):
        # Step 1: Conv1 and Residual1
        residual1 = self.conv1(x)
        x = self.norm1(residual1)

        # Step 2: E-MHSA
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> (h w) b c')
        k = self.avgpool(rearrange(x, '(h w) b c -> b c h w', h=h, w=w))  # Avg Pooling on K, V
        k = rearrange(k, 'b c h w -> (h w) b c')
        x, _ = self.e_mhsa(x, k, k)
        x = rearrange(x, '(h w) b c -> b c h w', h=h, w=w)
        residual2 = x + residual1

        # Step 3: Conv2 and Residual3
        x = self.conv2(residual2)
        residual3 = self.mhca(x) + x

        # Step 4: Concatenate Residual2 and Residual3, then MLP
        x = torch.cat([residual2, residual3], dim=1)
        x = self.norm3(x)
        x = self.mlp(x) + residual2

        return x

class NextViT(nn.Module):
    def __init__(self, num_classes=1, depths=[2, 2, 6, 2], head_dim=32):
        super(NextViT, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1280, 640, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
        )

        self.features = nn.ModuleList()
        input_channels = 640
        for i, depth in enumerate(depths):
            stage = []
            for j in range(depth):
                if i == len(depths) - 1 and j == depth - 1:  # 마지막 스테이지에 NTB 추가
                    stage.append(NTB(input_channels, input_channels, head_dim=head_dim))
                else:
                    stage.append(NCB(input_channels, input_channels, head_dim=head_dim))
            self.features.append(nn.Sequential(*stage))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.features:
            x = stage(x)
        x = self.head(x)
        return x

class EfficientNextViT(nn.Module):
    def __init__(self, config, selected_efficient_net=0):
        super(EfficientNextViT, self).__init__()

        if selected_efficient_net == 0:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7')

        for param in self.efficient_net.parameters():
            param.requires_grad = False
        for block in self.efficient_net._blocks[-3:]:
            for param in block.parameters():
                param.requires_grad = True

        num_classes = config['model']['num-classes']
        self.next_vit = NextViT(num_classes=num_classes)

    def forward(self, x):
        x = self.efficient_net.extract_features(x)
        x = self.next_vit(x)
        return x