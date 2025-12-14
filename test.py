import torch

checkpoint = torch.load(r"opencood\logs\point_pillar_intermediate_V2VAM_nocompressionV2_2025_11_14_21_14_17\net_epoch1.pth", map_location="cpu")
print(type(checkpoint))
print(checkpoint.keys())
print("Epoch:", checkpoint.get("epoch"))
print("Loss:", checkpoint.get("loss"))
