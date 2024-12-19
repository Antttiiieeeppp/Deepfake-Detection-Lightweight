import torch
from torch import nn
from einops import rearrange
from efficientnet_pytorch import EfficientNet
import cv2
import re
import numpy as np
from torch import einsum
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class EfficientViT(nn.Module):
    def __init__(self, config, channels=1280, selected_efficient_net=0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image_size = config["model"]["image-size"]
        patch_size = config["model"]["patch-size"]
        num_classes = config["model"]["num-classes"]
        dim = config["model"]["dim"]
        depth = config["model"]["depth"]
        heads = config["model"]["heads"]
        mlp_dim = config["model"]["mlp-dim"]
        emb_dim = config["model"]["emb-dim"]
        dim_head = config["model"]["dim-head"]
        dropout = config["model"]["dropout"]
        emb_dropout = config["model"]["emb-dropout"]

        assert (
            image_size % patch_size == 0
        ), "image dimensions must be divisible by the patch size"

        self.selected_efficient_net = selected_efficient_net

        if selected_efficient_net == 0:
            print("Loading EfficientNet-B0 pretrained model...")
            self.efficient_net = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            print("Loading EfficientNet-B7 pretrained model...")
            self.efficient_net = EfficientNet.from_pretrained("efficientnet-b7")
            checkpoint = torch.load(
                "weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
                map_location="cpu",
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.efficient_net.load_state_dict(
                {re.sub("^module.", "", k): v for k, v in state_dict.items()},
                strict=False,
            )

        for i in range(0, len(self.efficient_net._blocks)):
            for param in self.efficient_net._blocks[i].parameters():
                if i >= len(self.efficient_net._blocks) - 3:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        num_patches = (image_size // patch_size) ** 2
        # patch_dim은 pruning 후에도 conv_head out_channels가 1280이므로 그대로 유지
        patch_dim = channels

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(
            patch_dim, dim
        )  # 항상 1280을 입력받는다고 가정
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, num_classes)
        )

        print("EfficientViT Initialized with patch_size:", patch_size, ", dim:", dim)
        self.to(self.device)

    def check_channel_consistency(self):
        print("\n[Check] Checking Channel Consistency...")

        conv_stem_out = self.efficient_net._conv_stem.out_channels
        bn0_out = self.efficient_net._bn0.num_features
        assert (
            conv_stem_out == bn0_out
        ), f"Mismatch: _conv_stem out_channels ({conv_stem_out}) != _bn0 num_features ({bn0_out})"

        for idx, block in enumerate(self.efficient_net._blocks):
            if hasattr(block, "_project_conv") and hasattr(block, "_bn2"):
                proj_out = block._project_conv.out_channels
                bn2_out = block._bn2.num_features
                assert (
                    proj_out == bn2_out
                ), f"Mismatch in Block {idx}: _project_conv out_channels ({proj_out}) != _bn2 num_features ({bn2_out})"

        print("[Check] All channels are consistent!\n")

    def structured_prune_efficientnet(self, layer_indices, prune_ratio=0.5):
        print("[Info] Pruning EfficientNet with Structured Pruning...")
        print("  Target Layers:", layer_indices, "Prune Ratio:", prune_ratio)

        prev_out_channels = None

        for layer_idx in layer_indices:
            layer = self.efficient_net._blocks[layer_idx]

            if not (hasattr(layer, "_expand_conv") and hasattr(layer, "_project_conv")):
                print(
                    f"  Layer {layer_idx}: No expandable structure found, skipping pruning."
                )
                continue

            block_args = layer._block_args
            has_skip = (
                block_args.stride == 1
                and block_args.input_filters == block_args.output_filters
            )

            expand_conv = layer._expand_conv
            exp_weights = expand_conv.weight.data.cpu().numpy()
            exp_out_channels = exp_weights.shape[0]
            exp_in_channels = exp_weights.shape[1]

            if prev_out_channels is not None and exp_in_channels != prev_out_channels:
                print(
                    f"Warning: Input channels mismatch. Expected {prev_out_channels}, got {exp_in_channels}"
                )
                continue

            num_keep_exp = int(exp_out_channels * (1 - prune_ratio))
            l1_norms = np.sum(np.abs(exp_weights), axis=(1, 2, 3))
            important_channels = np.argsort(l1_norms)[-num_keep_exp:]
            important_channels.sort()

            exp_pruned_weight = exp_weights[important_channels, :, :, :]
            new_expand_conv = nn.Conv2d(
                in_channels=exp_in_channels,
                out_channels=num_keep_exp,
                kernel_size=expand_conv.kernel_size,
                stride=expand_conv.stride,
                padding=expand_conv.padding,
                bias=False,
            ).to(self.device)
            new_expand_conv.weight = nn.Parameter(
                torch.tensor(exp_pruned_weight).to(self.device)
            )
            layer._expand_conv = new_expand_conv

            if hasattr(layer, "_bn0"):
                bn0 = layer._bn0
                new_bn0 = nn.BatchNorm2d(num_keep_exp).to(self.device)
                with torch.no_grad():
                    new_bn0.weight = nn.Parameter(
                        bn0.weight[important_channels].clone()
                    )
                    new_bn0.bias = nn.Parameter(bn0.bias[important_channels].clone())
                    new_bn0.running_mean = bn0.running_mean[important_channels].clone()
                    new_bn0.running_var = bn0.running_var[important_channels].clone()
                layer._bn0 = new_bn0

            dw_conv = layer._depthwise_conv
            dw_weights = dw_conv.weight.data.cpu().numpy()
            new_dw_conv = nn.Conv2d(
                in_channels=num_keep_exp,
                out_channels=num_keep_exp,
                kernel_size=dw_conv.kernel_size,
                stride=dw_conv.stride,
                padding=dw_conv.padding,
                groups=num_keep_exp,
                bias=False,
            ).to(self.device)
            new_dw_conv.weight = nn.Parameter(
                torch.tensor(dw_weights[important_channels]).to(self.device)
            )
            layer._depthwise_conv = new_dw_conv

            bn1 = layer._bn1
            new_bn1 = nn.BatchNorm2d(num_keep_exp).to(self.device)
            with torch.no_grad():
                new_bn1.weight = nn.Parameter(bn1.weight[important_channels].clone())
                new_bn1.bias = nn.Parameter(bn1.bias[important_channels].clone())
                new_bn1.running_mean = bn1.running_mean[important_channels].clone()
                new_bn1.running_var = bn1.running_var[important_channels].clone()
            layer._bn1 = new_bn1

            proj_conv = layer._project_conv
            proj_weights = proj_conv.weight.data.cpu().numpy()
            out_channels = proj_weights.shape[0]

            if has_skip:
                num_keep_proj = out_channels
                proj_pruned_weight = proj_weights[:, important_channels, :, :]
            else:
                num_keep_proj = int(out_channels * (1 - prune_ratio))
                proj_pruned_weight = proj_weights[
                    :num_keep_proj, important_channels, :, :
                ]

            new_proj_conv = nn.Conv2d(
                in_channels=num_keep_exp,
                out_channels=num_keep_proj,
                kernel_size=proj_conv.kernel_size,
                stride=proj_conv.stride,
                padding=proj_conv.padding,
                bias=False,
            ).to(self.device)
            new_proj_conv.weight = nn.Parameter(
                torch.tensor(proj_pruned_weight).to(self.device)
            )
            layer._project_conv = new_proj_conv

            bn2 = layer._bn2
            new_bn2 = nn.BatchNorm2d(num_keep_proj).to(self.device)
            with torch.no_grad():
                new_bn2.weight = nn.Parameter(bn2.weight[:num_keep_proj].clone())
                new_bn2.bias = nn.Parameter(bn2.bias[:num_keep_proj].clone())
                new_bn2.running_mean = bn2.running_mean[:num_keep_proj].clone()
                new_bn2.running_var = bn2.running_var[:num_keep_proj].clone()
            layer._bn2 = new_bn2

            if hasattr(layer, "_se_reduce") and hasattr(layer, "_se_expand"):
                se_ratio = layer._se_ratio if hasattr(layer, "_se_ratio") else 0.25
                num_reduced_filters = max(1, int(num_keep_exp * se_ratio))

                old_se_reduce = layer._se_reduce
                old_se_expand = layer._se_expand

                new_se_reduce = nn.Conv2d(
                    num_keep_exp, num_reduced_filters, kernel_size=1, bias=True
                ).to(self.device)
                new_se_expand = nn.Conv2d(
                    num_reduced_filters, num_keep_exp, kernel_size=1, bias=True
                ).to(self.device)

                with torch.no_grad():
                    new_se_reduce.weight.data = old_se_reduce.weight.data[
                        :, important_channels, :, :
                    ]
                    new_se_reduce.bias.data = old_se_reduce.bias.data

                    new_se_expand.weight.data = old_se_expand.weight.data[
                        important_channels, :, :, :
                    ]
                    new_se_expand.bias.data = old_se_expand.bias.data[
                        important_channels
                    ]

                layer._se_reduce = new_se_reduce
                layer._se_expand = new_se_expand

            prev_out_channels = num_keep_proj

            print(
                f"  Layer {layer_idx}: Pruned expand from {exp_out_channels} to {num_keep_exp} channels, "
                f"project from {out_channels} to {num_keep_proj} channels."
            )

        if prev_out_channels is not None:
            conv_head = self.efficient_net._conv_head
            conv_head_weights = conv_head.weight.data.cpu().numpy()
            conv_head_bias = (
                conv_head.bias.data.cpu().numpy()
                if conv_head.bias is not None
                else None
            )

            # 여기서 out_channels는 그대로 유지 (기본 1280) -> final dimension 유지
            # 단지 in_channels만 맞춰줍니다.
            new_conv_head = nn.Conv2d(
                in_channels=prev_out_channels,
                out_channels=conv_head_weights.shape[0],  # 1280 그대로 유지
                kernel_size=conv_head.kernel_size,
                stride=conv_head.stride,
                padding=conv_head.padding,
                bias=conv_head.bias is not None,
            ).to(self.device)

            with torch.no_grad():
                new_conv_head.weight.data = torch.tensor(
                    conv_head_weights[:, :prev_out_channels, :, :]
                ).to(self.device)
                if conv_head_bias is not None:
                    new_conv_head.bias.data = torch.tensor(conv_head_bias).to(
                        self.device
                    )

            self.efficient_net._conv_head = new_conv_head
            print(
                f"  Conv head input channels adjusted from {conv_head_weights.shape[1]} to {prev_out_channels}"
            )
            # patch_to_embedding는 재설정하지 않음 (계속 1280 입력)

        self.check_channel_consistency()
        print("[Info] Structured Pruning Complete!\n")

    def forward(self, img, mask=None):
        img = img.to(self.device)

        x = self.efficient_net._conv_stem(img)
        x = self.efficient_net._bn0(x)
        x = self.efficient_net._swish(x)

        for idx, block in enumerate(self.efficient_net._blocks):
            drop_connect_rate = None
            if idx < len(self.efficient_net._blocks) - 1:
                drop_connect_rate = self.efficient_net._global_params.drop_connect_rate
                drop_connect_rate *= float(idx) / len(self.efficient_net._blocks)

            if hasattr(block, "_block_args") and block._block_args.stride == 1:
                identity = x
                if hasattr(block, "_expand_conv"):
                    x = block._expand_conv(x)
                    x = block._bn0(x)
                    x = block._swish(x)

                    dw_conv = block._depthwise_conv
                    kernel_size = dw_conv.kernel_size[0]
                    if kernel_size > x.shape[-1]:
                        padding_size = (kernel_size - 1) // 2
                        x = F.pad(
                            x, (padding_size, padding_size, padding_size, padding_size)
                        )

                    x = dw_conv(x)
                    x = block._bn1(x)
                    x = block._swish(x)

                    if hasattr(block, "_se_reduce"):
                        x_squeezed = F.adaptive_avg_pool2d(x, 1)
                        x_squeezed = block._se_reduce(x_squeezed)
                        x_squeezed = block._swish(x_squeezed)
                        x_squeezed = block._se_expand(x_squeezed)
                        x = torch.sigmoid(x_squeezed) * x

                    x = block._project_conv(x)
                    x = block._bn2(x)

                    if x.shape != identity.shape:
                        if x.shape[-2:] != identity.shape[-2:]:
                            identity = F.adaptive_avg_pool2d(identity, x.shape[-2:])
                        if x.shape[1] != identity.shape[1]:
                            identity = F.pad(
                                identity,
                                (0, 0, 0, 0, 0, x.shape[1] - identity.shape[1]),
                            )

                    x = x + identity
            else:
                x = block(x, drop_connect_rate=drop_connect_rate)

        x = self.efficient_net._conv_head(x)
        x = self.efficient_net._bn1(x)
        x = self.efficient_net._swish(x)

        # 최종 out_channels는 pruning 전과 동일하게 1280으로 유지됨
        x = F.adaptive_avg_pool2d(x, output_size=7)

        y = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=1, p2=1)
        # y shape: b, 49, 1280
        y = self.patch_to_embedding(y)  # 이제 (N,1280) * (1280,1024) 계산 가능

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)

        x += self.pos_embedding[:, : x.shape[1]]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])

        x = self.mlp_head(x)
        return x
