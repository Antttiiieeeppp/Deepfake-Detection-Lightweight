import torch
from jh_efficient_vit import EfficientViT

# 파라미터 수 계산 함수
def count_parameters(model):
    if isinstance(model, torch.nn.Parameter):
        return model.numel()  # 개별 Parameter 객체 처리
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 모델 초기화
config = {
    'model': {
        'image-size': 224,
        'patch-size': 1,
        'num-classes': 1000,
        'dim': 512,
        'depth': 6,
        'heads': 8,
        'mlp-dim': 1024,
        'emb-dim': 256,
        'dim-head': 64,
        'dropout': 0.1,
        'emb-dropout': 0.1
    }
}

# EfficientViT 모델 생성
model = EfficientViT(config=config, channels=512, selected_efficient_net=0)

# EfficientNet 부분 파라미터 수
efficient_net_params = count_parameters(model.efficient_net)

# ViT 부분 파라미터 수 (Parameter 객체도 처리)
vit_params = count_parameters(model.transformer) + \
             count_parameters(model.pos_embedding) + \
             count_parameters(model.patch_to_embedding) + \
             count_parameters(model.cls_token) + \
             count_parameters(model.mlp_head)

# 전체 파라미터 수
total_params = count_parameters(model)

# 결과 출력
print(f"EfficientNet 파라미터 수: {efficient_net_params}")
print(f"ViT 파라미터 수: {vit_params}")
print(f"모델 총 파라미터 수: {total_params}")