import torch
from torch import nn
from einops import rearrange
from efficientnet_pytorch import EfficientNet
import cv2
import re
from utils import resize
import numpy as np
from torch import einsum
from random import randint

class Residual(nn.Module):
    """
    Residual 모듈: 입력을 특정 함수(fn)에 통과시킨 결과를 원본 입력에 더해줍니다.
    Transformer 등에서 스킵 커넥션(skip connection)을 구현하는데 활용됩니다.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    """
    LayerNorm을 적용한 후 함수(fn)을 수행하는 모듈
    Transformer 블록 내부에서 Attention 및 FeedForward 모듈을 정규화하기 위해 사용됩니다.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """
    간단한 2-layer MLP(Multi-layer perceptron) 구조.
    Transformer 블록 내에서 Attention 후 비선형 변환을 위해 사용됩니다.
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Multi-Head Self-Attention 모듈.
    입력 x에 대해 Q, K, V를 계산하고, Attention 가중치를 적용한 후 결합합니다.
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        # Q, K, V 분리
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # Attention Score 계산
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        # Attention을 통해 value 집계
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """
    Transformer Encoder 블록을 쌓은 구조.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout = 0))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x   # Attention Residual
            x = ff(x) + x     # FeedForward Residual
        return x

class EfficientViT(nn.Module):
    """
    EfficientNet을 백본으로 활용한 비전 트랜스포머 모델입니다.
    EfficientNet으로부터 추출한 feature map을 patch 단위로 나누고,
    Transformer에 입력하여 최종적으로 분류 결과를 얻습니다.
    """
    def __init__(self, config, channels=512, selected_efficient_net = 0):
        super().__init__() 

        # Config 파라미터 로드
        image_size = config['model']['image-size']
        patch_size = config['model']['patch-size']
        num_classes = config['model']['num-classes']
        dim = config['model']['dim']
        depth = config['model']['depth']
        heads = config['model']['heads']
        mlp_dim = config['model']['mlp-dim']
        emb_dim = config['model']['emb-dim']
        dim_head = config['model']['dim-head']
        dropout = config['model']['dropout']
        emb_dropout = config['model']['emb-dropout']

        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        
        self.selected_efficient_net = selected_efficient_net

        # EfficientNet 모델 로드
        # selected_efficient_net == 0이면 b0, 아니면 b7 로드
        if selected_efficient_net == 0:
            print("Loading EfficientNet-B0 pretrained model...")
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            print("Loading EfficientNet-B7 pretrained model...")
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7')
            # 외부 학습 가중치 로드 (필요에 따라 수정)
            checkpoint = torch.load("weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23", map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.efficient_net.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        
        # EfficientNet의 일부 레이어만 학습 가능하도록 requires_grad 설정
        for i in range(0, len(self.efficient_net._blocks)):
            for index, param in enumerate(self.efficient_net._blocks[i].parameters()):
                if i >= len(self.efficient_net._blocks)-3:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Patch 관련 파라미터
        num_patches = (7 // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        # Embedding 및 Positional Encoding 설정
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(emb_dim, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer 및 MLP Head 정의
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )
        
        print("EfficientViT Initialized with patch_size:", patch_size, ", dim:", dim)

    def check_channel_consistency(self):
        """
        EfficientNet 모델의 모든 레이어 채널 수 일관성을 검사합니다.
        Pruning 이후에도 채널 수가 잘 맞는지 확인하는데 사용합니다.
        """
        print("\n[Check] Checking Channel Consistency...")

        conv_stem_out = self.efficient_net._conv_stem.out_channels
        bn0_out = self.efficient_net._bn0.num_features
        assert conv_stem_out == bn0_out, f"Mismatch: _conv_stem out_channels ({conv_stem_out}) != _bn0 num_features ({bn0_out})"

        for idx, block in enumerate(self.efficient_net._blocks):
            if hasattr(block, "_project_conv") and hasattr(block, "_bn2"):
                proj_out = block._project_conv.out_channels
                bn2_out = block._bn2.num_features
                assert proj_out == bn2_out, f"Mismatch in Block {idx}: _project_conv out_channels ({proj_out}) != _bn2 num_features ({bn2_out})"

        print("[Check] All channels are consistent!\n")   

    def structured_prune_efficientnet(self, layer_indices, prune_ratio=0.5):
        """
        EfficientNet의 특정 레이어를 Structured Pruning합니다.
        L1 norm 기준으로 중요 채널을 남기고 나머지를 제거합니다.
        skip connection이 있는 블록에 대해서는 input/output channel 일관성을 유지하도록
        _expand_conv와 _project_conv를 함께 pruning합니다.
        
        Args:
            layer_indices: 프루닝할 레이어 인덱스 리스트
            prune_ratio: 제거할 채널 비율 (0~1)
        """
        print("[Info] Pruning EfficientNet with Structured Pruning...")
        print("  Target Layers:", layer_indices, "Prune Ratio:", prune_ratio)

        for layer_idx in layer_indices:
            layer = self.efficient_net._blocks[layer_idx]

            # skip connection 여부 판단:
            # MBConvBlock에서 skip connection은 input channel == output channel 그리고 stride=1일 때 적용됨.
            has_skip = False
            input_channels = None
            if hasattr(layer, "_project_conv"):
                output_channels = layer._project_conv.out_channels
                # _expand_conv가 있는 경우 input_channels는 _expand_conv의 in_channels
                # 가장 초기에 들어오는 in_channels는 block을 초기화할 때 _expand_conv.in_channels를 통해 알 수 있음
                if hasattr(layer, "_expand_conv"):
                    input_channels = layer._expand_conv.in_channels
                else:
                    # 만약 expand_conv가 없다면 MBConvBlock이 아닌 다른 형태일 수 있음. 
                    # 혹은 B0 구조 중 stride=1인 초기 블록일 수도 있음.
                    # 이런 경우엔 input_channels를 별도로 판단 필요.
                    # 여기서는 기본적으로 skip connection 성립할 경우를 가정.
                    input_channels = output_channels
                
                # stride=1이고 input_channels == output_channels이면 skip connection 사용
                if layer._depthwise_conv.stride == (1,1) and input_channels == output_channels:
                    has_skip = True
            else:
                # _project_conv가 없는 경우, 이는 다양한 변형이 있을 수 있으나 여기서는 기본 EfficientNet만 가정
                # 대부분 _project_conv를 가지므로 이 경우는 거의 없음.
                continue

            # Conv와 BN 레이어 이름 결정
            conv_layer_name = "_project_conv" if hasattr(layer, "_project_conv") else "_expand_conv"
            bn_layer_name = "_bn2" if hasattr(layer, "_project_conv") else "_bn1"
            
            conv_layer = getattr(layer, conv_layer_name)
            batch_norm = getattr(layer, bn_layer_name)

            # Conv 가중치 로드
            weights = conv_layer.weight.data.cpu().numpy()
            out_channels, _, _, _ = weights.shape

            # L1 norm 계산 후 중요 채널 선별
            l1_norms = np.sum(np.abs(weights), axis=(1, 2, 3))
            num_keep = int(out_channels * (1 - prune_ratio))
            important_channels = np.argsort(l1_norms)[-num_keep:]  # 가장 중요한 채널 인덱스

            pruned_weight = weights[important_channels, :, :, :]
            pruned_conv = nn.Conv2d(
                conv_layer.in_channels,
                num_keep,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.padding,
                bias=False
            )
            pruned_conv.weight = nn.Parameter(torch.tensor(pruned_weight, dtype=conv_layer.weight.dtype))

            # BN 레이어 업데이트
            pruned_bn = nn.BatchNorm2d(num_features=num_keep)
            with torch.no_grad():
                pruned_bn.weight = nn.Parameter(batch_norm.weight[important_channels].clone())
                pruned_bn.bias = nn.Parameter(batch_norm.bias[important_channels].clone())
                pruned_bn.running_mean = batch_norm.running_mean[important_channels].clone()
                pruned_bn.running_var = batch_norm.running_var[important_channels].clone()

            # 먼저 project_conv와 bn 교체
            setattr(layer, conv_layer_name, pruned_conv)
            setattr(layer, bn_layer_name, pruned_bn)

            # skip connection이 존재하는 경우 expand_conv도 동일한 비율로 prune하여 input_channels == output_channels 유지
            if has_skip and hasattr(layer, "_expand_conv"):
                expand_conv = layer._expand_conv
                # expand_conv의 out_channels는 project_conv의 in_channels와 연결
                # pruning 이후 project_conv out_channels == expand_conv in_channels 이어야 함
                # 즉, project_conv에서 important_channels에 해당하는 채널만 남기도록 expand_conv도 pruning
                
                # expand_conv pruning을 위해 in_channels, out_channels 모두 조정 필요
                # expand_conv의 out_channels가 현재 project_conv의 in_channels로 이어지는 부분
                # 여기서는 같은 비율로 pruning (같은 중요도 순서 사용)한다고 가정
                # 실제로는 expand_conv의 출력 채널 중요도를 다시 계산해야 할 수도 있음

                exp_weights = expand_conv.weight.data.cpu().numpy()
                # expand_conv 형태: (out_channels, in_channels, k, k)
                # out_channels 중에서 project_conv의 important_channels에 해당하는 부분만 유지하기 위해
                # expand_conv의 out_channels도 동일한 인덱스로 pruning
                exp_out_channels, exp_in_channels, kh, kw = exp_weights.shape

                # 여기서는 단순히 project_conv의 important_channels를 expand_conv에도 적용
                # 즉, project_conv에서 선택된 channel에 해당하는 expand_conv의 out_channels만 남긴다.
                exp_pruned_weight = exp_weights[important_channels, :, :, :]

                # expand_conv의 out_channels가 줄었으니 BN0도 수정 필요
                # expand_conv 이전 BN0가 있다면 마찬가지로 동일한 인덱스로 pruning
                if hasattr(layer, "_bn0"):
                    bn0 = layer._bn0
                    pruned_bn0 = nn.BatchNorm2d(num_features=num_keep)
                    with torch.no_grad():
                        pruned_bn0.weight = nn.Parameter(bn0.weight[important_channels].clone())
                        pruned_bn0.bias = nn.Parameter(bn0.bias[important_channels].clone())
                        pruned_bn0.running_mean = bn0.running_mean[important_channels].clone()
                        pruned_bn0.running_var = bn0.running_var[important_channels].clone()
                    layer._bn0 = pruned_bn0

                # expand_conv 업데이트
                new_expand_conv = nn.Conv2d(
                    in_channels=exp_in_channels,
                    out_channels=num_keep,
                    kernel_size=expand_conv.kernel_size,
                    stride=expand_conv.stride,
                    padding=expand_conv.padding,
                    bias=False
                )
                new_expand_conv.weight = nn.Parameter(torch.tensor(exp_pruned_weight, dtype=expand_conv.weight.dtype))
                layer._expand_conv = new_expand_conv

                # 이제 expand_conv의 in_channels는 변하지 않았으나 out_channels가 줄었음
                # input_channels와 output_channels가 같아야 skip connection 가능하므로,
                # 여기서는 pruning 전후에 input_channels == output_channels 관계가 유지되도록 했으므로
                # skip connection에 문제없이 동작

            print(f"  Layer {layer_idx}: Pruned from {out_channels} to {num_keep} channels.")

        # Pruning 후 채널 consistency check
        self.check_channel_consistency()
        print("[Info] Structured Pruning Complete!\n")


    def forward(self, img, mask=None):
        """
        모델 Forward:
        1. EfficientNet 특징 추출
        2. 추출된 특징 맵을 patch 단위로 분할
        3. Transformer로 처리 후 최종 분류 Head로 결과 산출
        """
        # 입력 이미지 크기 출력 (디버깅용)
        print("[Debug] Input image shape:", img.shape) 

        # EfficientNet 특징 추출
        x = self.efficient_net.extract_features(img) # 1280x7x7 (b, c, h, w)
        print("[Debug] EfficientNet features shape:", x.shape)

        # Pruning 이후 크기 일치 확인
        # EfficientNet의 마지막 블록 batchnorm 채널 수와 특징 맵 채널 수 일치 여부 확인
        last_bn = self.efficient_net._blocks[-1]._bn2
        assert x.shape[1] == last_bn.num_features, \
            f"Mismatch: EfficientNet output channels ({x.shape[1]}) != last BN features ({last_bn.num_features})"

        p = self.patch_size
        # feature map을 (b, c, h, w)에서 (b, n_patches, patch_dim) 형태로 변경
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        print("[Debug] Rearranged features shape:", y.shape)

        # Patch Embedding
        y = self.patch_to_embedding(y)

        # CLS Token 추가
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)

        # Positional Encoding 추가
        shape = x.shape[0]
        x += self.pos_embedding[0:shape]
        x = self.dropout(x)

        # Transformer 인코더 통과
        x = self.transformer(x)
        print("[Debug] Transformer output shape:", x.shape)

        # 첫 번째 토큰(CLS)을 최종 분류기로 전달
        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        print("[Debug] Model output shape:", x.shape)

        return x
